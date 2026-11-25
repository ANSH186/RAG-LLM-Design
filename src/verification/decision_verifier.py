"""Independent decision verifier.

This module deliberately does not call the main decision engine or offer normalizer.
It re-derives the applicable policy, re-parses the source amount, recalculates the
concession level and approval check, independently applies precedence, and compares
the resulting label with the decision under review.
"""
from __future__ import annotations

import re
from decimal import Decimal

from src.decision.security_rules import detect_banned_terms, detect_prompt_injection
from src.evidence.evidence_schema import EvidencePackage
from src.evidence.source_registry import SourceRegistry
from src.extraction.deterministic_extractor import (
    detect_billing_period,
    detect_gst,
    detect_unit_factor,
    find_money,
)
from src.extraction.extraction_schema import NormalizedOffer, OfferExtraction
from src.knowledge.policy_snapshot import PolicySnapshot
from src.knowledge.schema import CompiledKnowledge, SupplierStatus
from src.utils.hashing import sha256_json
from src.utils.numbers import round_half_up
from src.utils.text import latest_supplier_body
from src.verification.consistency_checker import consistency_checks
from src.verification.evidence_verifier import verify_evidence_package
from src.verification.schemas import (
    VerificationCheck,
    VerificationIssue,
    VerificationResult,
    VerificationStatus,
)


def _add(issues: list[VerificationIssue], code: str, detail: str, **kwargs) -> None:
    issues.append(VerificationIssue(code=code, detail=detail, **kwargs))


def _verify_snapshot(
    knowledge: CompiledKnowledge, snapshot: PolicySnapshot, issues: list[VerificationIssue]
) -> None:
    category = knowledge.categories.get(snapshot.category_slug)
    if category is None:
        _add(issues, "unknown_category", "snapshot category does not exist in the playbook")
        return
    supplier = knowledge.suppliers.get(snapshot.supplier_id)
    if supplier is None:
        _add(issues, "unknown_supplier", "snapshot supplier does not exist in the playbook")
        return
    if supplier.name != snapshot.supplier_name:
        _add(issues, "supplier_name_mismatch", "snapshot supplier name differs from the source doc")
    expected = {
        "target_price": category.target_price,
        "walkaway_price": category.walkaway_price,
        "approval_limit": category.approval_limit,
        "gst_rate": category.gst_rate,
        "negotiation_rounds": category.negotiation_rounds,
        "concession_beta": category.concession_beta,
    }
    applied: list[str] = []
    for amendment in knowledge.amendments:
        if amendment.category_slug != snapshot.category_slug:
            continue
        if amendment.effective_from > snapshot.negotiation_date:
            continue
        applied.append(amendment.amendment_id)
        expected[amendment.field] = (
            int(amendment.value) if amendment.field == "negotiation_rounds" else amendment.value
        )
    cap = None
    for price_cap in supplier.price_caps:
        if price_cap.category_slug == snapshot.category_slug:
            cap = price_cap.value
            expected["walkaway_price"] = price_cap.value
    checks = {
        "target_price": expected["target_price"],
        "walkaway_price": expected["walkaway_price"],
        "approval_limit": expected["approval_limit"],
        "gst_rate": expected["gst_rate"],
        "negotiation_rounds": expected["negotiation_rounds"],
        "concession_beta": expected["concession_beta"],
    }
    actual = {
        "target_price": snapshot.target_price,
        "walkaway_price": snapshot.walkaway_price,
        "approval_limit": snapshot.approval_limit,
        "gst_rate": snapshot.gst_rate,
        "negotiation_rounds": snapshot.negotiation_rounds,
        "concession_beta": snapshot.concession_beta,
    }
    for field, expected_value in checks.items():
        if Decimal(str(actual[field])) != Decimal(str(expected_value)):
            _add(
                issues,
                "snapshot_value_mismatch",
                f"{field} does not match a fresh temporal resolution",
                field=field,
                expected=str(expected_value),
                actual=str(actual[field]),
            )
    if tuple(applied) != snapshot.amendments_applied:
        _add(
            issues,
            "amendment_set_mismatch",
            "applied amendment list is not the full effective-date-filtered set",
            expected=",".join(applied),
            actual=",".join(snapshot.amendments_applied),
        )
    for amendment_id in snapshot.amendments_applied:
        amendment = next((a for a in knowledge.amendments if a.amendment_id == amendment_id), None)
        if amendment is None:
            _add(issues, "unknown_amendment", "snapshot applies an unknown amendment", actual=amendment_id)
        elif amendment.effective_from > snapshot.negotiation_date:
            _add(
                issues,
                "future_amendment_applied",
                "snapshot applies an amendment effective after the negotiation",
                actual=f"{amendment_id} {amendment.effective_from}",
            )
    if snapshot.supplier_status != supplier.status:
        _add(
            issues,
            "supplier_status_mismatch",
            "snapshot supplier status differs from the official supplier page",
            expected=supplier.status.value,
            actual=snapshot.supplier_status.value,
        )
    expected_overrides = {
        override.currency: override.inr_per_unit for override in supplier.exchange_rate_overrides
    }
    if {k: Decimal(str(v)) for k, v in expected_overrides.items()} != {
        k: Decimal(str(v)) for k, v in snapshot.supplier_exchange_overrides.items()
    }:
        _add(
            issues,
            "exchange_override_mismatch",
            "supplier exchange-rate overrides differ from the source contract",
        )
    if snapshot.supplier_price_cap != cap:
        _add(
            issues,
            "price_cap_mismatch",
            "supplier price cap differs from the source contract",
            expected=str(cap),
            actual=str(snapshot.supplier_price_cap),
        )
    for currency, rate in knowledge.currency_rates.items():
        if currency not in snapshot.currency_rates:
            _add(issues, "currency_table_incomplete", "base currency table missing a currency", actual=currency)
        elif Decimal(str(snapshot.currency_rates[currency])) != Decimal(str(rate.inr_per_unit)):
            _add(issues, "currency_table_mismatch", "base currency rate mismatch", actual=currency)
    for field in (
        "target_price",
        "walkaway_price",
        "approval_limit",
        "gst_rate",
        "negotiation_rounds",
        "concession_beta",
    ):
        if not snapshot.field_provenance.get(field):
            _add(issues, "missing_field_provenance", "policy field lacks provenance", field=field)
    # Recompute the policy version from the resolved snapshot values.
    payload = {
        "configuration_hash": knowledge.configuration_hash,
        "thread_id": snapshot.thread_id,
        "category": snapshot.category_slug,
        "supplier_id": snapshot.supplier_id,
        "date": snapshot.negotiation_date.isoformat(),
        "values": {
            "target_price": str(snapshot.target_price),
            "walkaway_price": str(snapshot.walkaway_price),
            "approval_limit": str(snapshot.approval_limit),
            "gst_rate": str(snapshot.gst_rate),
            "negotiation_rounds": str(snapshot.negotiation_rounds),
            "concession_beta": str(snapshot.concession_beta),
        },
        "amendments": list(snapshot.amendments_applied),
        "cap": str(snapshot.supplier_price_cap) if snapshot.supplier_price_cap is not None else None,
        "overrides": {k: str(v) for k, v in snapshot.supplier_exchange_overrides.items()},
    }
    expected_hash = sha256_json(payload)[:12]
    if not snapshot.policy_version.endswith(expected_hash):
        _add(
            issues,
            "policy_version_hash_mismatch",
            "policy version hash was not derived from the resolved snapshot",
            expected=expected_hash,
            actual=snapshot.policy_version,
        )


def _verify_extraction_grounding(
    extraction: OfferExtraction,
    snapshot: PolicySnapshot,
    knowledge: CompiledKnowledge,
    issues: list[VerificationIssue],
) -> None:
    latest = latest_supplier_body(extraction.source_message)
    if extraction.latest_message_text != latest:
        _add(issues, "latest_message_segmentation_mismatch", "latest supplier text was not segmented correctly")
    if extraction.offer_text:
        if extraction.offer_text not in extraction.source_message:
            _add(issues, "offer_text_not_verbatim", "offer sentence is absent from the source")
        scan_text = extraction.offer_text
        if extraction.is_correction:
            cut = re.search(r"(?i),\s*not\b", scan_text)
            if cut:
                scan_text = scan_text[: cut.start()]
        reparsed = find_money(scan_text)
        expected_candidates = [(c.amount, c.currency) for c in reparsed]
        actual_candidates = [(c.amount, c.currency) for c in extraction.price_candidates]
        if expected_candidates != actual_candidates:
            _add(
                issues,
                "price_candidate_mismatch",
                "re-parsing the offer sentence produced different price candidates",
                expected=str(expected_candidates),
                actual=str(actual_candidates),
            )
        if detect_gst(extraction.offer_text) != extraction.gst_treatment:
            _add(
                issues,
                "gst_detection_mismatch",
                "re-parsed GST treatment differs from extraction",
                expected=detect_gst(extraction.offer_text),
                actual=extraction.gst_treatment,
            )
        if detect_billing_period(extraction.offer_text) != extraction.billing_period:
            _add(
                issues,
                "billing_period_detection_mismatch",
                "re-parsed billing period differs from extraction",
                expected=str(detect_billing_period(extraction.offer_text)),
                actual=str(extraction.billing_period),
            )
        factor, label, _warning = detect_unit_factor(extraction.offer_text)
        category = knowledge.categories[snapshot.category_slug]
        if label and factor == 1:
            base_label = label.split()[0] if label.startswith("pack of") else label
            factor = category.subunit_factors.get(base_label, factor)
        if label == "ream":
            factor = category.subunit_factors.get("ream", factor)
        if factor != extraction.quoted_unit_factor:
            _add(
                issues,
                "unit_factor_mismatch",
                "re-parsed quoted unit factor differs from extraction",
                expected=str(factor),
                actual=str(extraction.quoted_unit_factor),
            )
    if extraction.previous_quote_candidates:
        if not extraction.previous_quote_text:
            _add(issues, "previous_quote_missing_text", "previous quote candidates lack source text")
        else:
            reparsed_prev = [(c.amount, c.currency) for c in find_money(extraction.previous_quote_text)]
            actual_prev = [(c.amount, c.currency) for c in extraction.previous_quote_candidates]
            if reparsed_prev != actual_prev:
                _add(
                    issues,
                    "previous_quote_candidate_mismatch",
                    "previous supplier quote re-parse differs",
                    expected=str(reparsed_prev),
                    actual=str(actual_prev),
                )


def _independent_period(
    value: Decimal,
    quoted: str | None,
    target: str | None,
    issues: list[VerificationIssue],
) -> tuple[Decimal | None, str | None]:
    if target is None and quoted is None:
        return value, None
    if target is None and quoted is not None:
        return None, "unexpected_billing_period"
    if target is not None and quoted is None:
        return None, "subscription_billing_period_missing"
    if quoted == target:
        return value, None
    if quoted == "month" and target == "year":
        return value * Decimal(12), None
    if quoted == "year" and target == "month":
        return value / Decimal(12), None
    return None, "unsupported_billing_period_conversion"


def _independent_rate(snapshot: PolicySnapshot, currency: str | None) -> Decimal | None:
    if currency is None:
        return None
    if currency == "INR":
        return Decimal(1)
    if currency in snapshot.supplier_exchange_overrides:
        return snapshot.supplier_exchange_overrides[currency]
    return snapshot.currency_rates.get(currency)


def _recompute_normalized(
    extraction: OfferExtraction,
    snapshot: PolicySnapshot,
    quantity: int,
    issues: list[VerificationIssue],
) -> tuple[Decimal | None, bool]:
    """Recompute normalized price independently from structured extraction fields."""
    if extraction.discount_base == "previous_quote":
        if not extraction.previous_quote_candidates:
            return None, True
        previous = extraction.previous_quote_candidates[0]
        value = previous.amount
        if extraction.previous_quote_unit_factor == 0:
            return None, True
        value = value / extraction.previous_quote_unit_factor
        value, _period_error = _independent_period(
            value, extraction.previous_quote_billing_period, snapshot.normalized_period, issues
        )
        if value is None:
            return None, True
        rate = _independent_rate(snapshot, previous.currency)
        if rate is None:
            return None, True
        value = value * rate
        if extraction.previous_quote_gst_treatment == "included":
            value = value / (Decimal(1) + snapshot.gst_rate)
        elif extraction.previous_quote_gst_treatment == "ambiguous":
            return None, True
        if extraction.discount_percent is None or extraction.discount_percent < 0 or extraction.discount_percent > 100:
            return None, True
        value = value * (Decimal(1) - extraction.discount_percent / Decimal(100))
        return value, False

    if not extraction.price_candidates:
        return None, True

    candidates = list(extraction.price_candidates)
    if extraction.selection_rule == "tier_by_quantity":
        if extraction.tier_threshold is None or len(candidates) < 2:
            return None, True
        candidates = [candidates[0] if quantity <= extraction.tier_threshold else candidates[1]]
    elif extraction.selection_rule == "single":
        if len(candidates) != 1:
            return None, True
    elif extraction.selection_rule == "correction":
        candidates = [candidates[0]]
    elif extraction.selection_rule == "range_high":
        pass
    else:
        return None, True

    computed: list[Decimal] = []
    for candidate in candidates:
        value = candidate.amount
        if extraction.discount_base == "list_price" and extraction.discount_percent is not None:
            if not (Decimal(0) <= extraction.discount_percent <= Decimal(100)):
                return None, True
            value = value * (Decimal(1) - extraction.discount_percent / Decimal(100))
        if extraction.volume_discount_percent is not None and extraction.volume_threshold is not None:
            if quantity >= extraction.volume_threshold:
                value = value * (Decimal(1) - extraction.volume_discount_percent / Decimal(100))
        if extraction.free_units is not None and extraction.paid_units is not None:
            if extraction.paid_units <= 0:
                return None, True
            value = value * Decimal(extraction.paid_units) / Decimal(
                extraction.paid_units + extraction.free_units
            )
        if extraction.quoted_unit_factor == 0:
            return None, True
        value = value / extraction.quoted_unit_factor
        if extraction.is_total:
            units = extraction.total_quantity or quantity
            if units <= 0:
                _add(issues, "total_quantity_invalid", "total offer has no valid quantity")
                return None, True
            value = value / Decimal(units)
        value, _period_error = _independent_period(
            value, extraction.billing_period, snapshot.normalized_period, issues
        )
        if value is None:
            return None, True
        rate = _independent_rate(snapshot, candidate.currency)
        if rate is None:
            return None, True
        value = value * rate
        if extraction.gst_treatment == "included":
            value = value / (Decimal(1) + snapshot.gst_rate)
        elif extraction.gst_treatment == "ambiguous":
            return None, True
        computed.append(value)
    if not computed:
        return None, True
    return max(computed) if extraction.selection_rule == "range_high" else computed[0], False


def _independent_level(snapshot: PolicySnapshot, round_number: int) -> Decimal:
    if snapshot.supplier_status == SupplierStatus.PROBATION:
        return snapshot.target_price
    if round_number == 1:
        return snapshot.target_price
    if round_number >= snapshot.negotiation_rounds:
        return snapshot.walkaway_price
    ratio = Decimal(round_number - 1) / Decimal(snapshot.negotiation_rounds - 1)
    return snapshot.target_price + (snapshot.walkaway_price - snapshot.target_price) * (
        ratio ** snapshot.concession_beta
    )


def _expected_outcome(
    *,
    snapshot: PolicySnapshot,
    extraction: OfferExtraction,
    normalized_price: Decimal | None,
    usable: bool,
    quantity: int,
    round_number: int,
    injections,
    banned,
) -> tuple[str, str, int | None, Decimal | None, Decimal | None]:
    """Return expected label, applied rule, counter, level, order value."""
    if not usable or normalized_price is None:
        return "ESCALATE:MISSING_INFO:", "MISSING_INFORMATION", None, None, None
    if snapshot.supplier_status == SupplierStatus.BLACKLISTED:
        return "REJECT:BLACKLISTED_SUPPLIER:", "BLACKLISTED_SUPPLIER", None, None, None
    if injections:
        return "ESCALATE:PROMPT_INJECTION:", "PROMPT_INJECTION", None, None, None
    if banned:
        return "ESCALATE:BANNED_TERM:", "BANNED_TERM", None, None, None
    if not (1 <= round_number <= snapshot.negotiation_rounds):
        return "ESCALATE:MISSING_INFO:", "MISSING_INFORMATION", None, None, None
    level = _independent_level(snapshot, round_number)
    if normalized_price <= level:
        total = normalized_price * Decimal(quantity)
        if total > snapshot.approval_limit:
            return "ESCALATE:OVER_APPROVAL_LIMIT:", "APPROVAL_LIMIT", None, level, total
        return "ACCEPT:ACCEPTABLE_PRICE:", "PRICE_DECISION", None, level, total
    if round_number >= snapshot.negotiation_rounds:
        return "REJECT:ABOVE_WALKAWAY:", "PRICE_DECISION", None, level, None
    return f"COUNTER:COUNTER_OFFER:{round_half_up(level)}", "PRICE_DECISION", round_half_up(level), level, None


def verify_decision(
    *,
    decision,
    extraction: OfferExtraction,
    normalized: NormalizedOffer,
    snapshot: PolicySnapshot,
    knowledge: CompiledKnowledge,
    evidence_package: EvidencePackage | None = None,
    registry: SourceRegistry | None = None,
    attempt: int = 1,
    corrected: bool = False,
) -> VerificationResult:
    issues: list[VerificationIssue] = []
    checks: list[VerificationCheck] = []
    registry = registry or SourceRegistry(knowledge)

    _verify_snapshot(knowledge, snapshot, issues)
    _verify_extraction_grounding(extraction, snapshot, knowledge, issues)
    independent_price, independent_unusable = _recompute_normalized(
        extraction, snapshot, decision.quantity, issues
    )
    if independent_unusable and independent_price is not None:
        independent_price = None
    if independent_price != normalized.normalized_price_inr_excl_gst:
        _add(
            issues,
            "normalized_price_mismatch",
            "independent normalization differs from the engine result",
            expected=str(independent_price),
            actual=str(normalized.normalized_price_inr_excl_gst),
        )
    expected_usable = not independent_unusable
    if expected_usable != normalized.is_usable:
        _add(
            issues,
            "usability_mismatch",
            "independent usability differs from normalization",
            expected=str(expected_usable),
            actual=str(normalized.is_usable),
        )

    injections = detect_prompt_injection(extraction.source_message)
    banned = detect_banned_terms(latest_supplier_body(extraction.source_message))
    if bool(injections) != bool(decision.injection_findings):
        _add(issues, "injection_findings_mismatch", "decision security findings differ from fresh scan")
    if bool(banned) != bool(decision.banned_term_findings):
        _add(issues, "banned_findings_mismatch", "decision banned-term findings differ from fresh scan")

    expected_label, expected_rule, expected_counter, expected_level, expected_order_value = _expected_outcome(
        snapshot=snapshot,
        extraction=extraction,
        normalized_price=independent_price,
        usable=expected_usable,
        quantity=decision.quantity,
        round_number=decision.round_number,
        injections=injections,
        banned=banned,
    )
    if expected_label != decision.label:
        _add(
            issues,
            "final_label_mismatch",
            "independent precedence evaluation produced a different final label",
            expected=expected_label,
            actual=decision.label,
        )
    if expected_rule != decision.applied_rule.value:
        _add(
            issues,
            "applied_rule_mismatch",
            "decision reports a different controlling rule",
            expected=expected_rule,
            actual=decision.applied_rule.value,
        )
    if expected_counter != decision.counter_price:
        _add(
            issues,
            "counter_price_mismatch",
            "counter price differs from independent concession rounding",
            expected=str(expected_counter),
            actual=str(decision.counter_price),
        )
    if expected_level is not None and decision.acceptable_level is not None:
        if expected_level != decision.acceptable_level:
            _add(
                issues,
                "acceptable_level_mismatch",
                "concession level differs from independent calculation",
                expected=str(expected_level),
                actual=str(decision.acceptable_level),
            )
    elif expected_level != decision.acceptable_level:
        _add(
            issues,
            "acceptable_level_presence_mismatch",
            "acceptable level presence differs from independent calculation",
            expected=str(expected_level),
            actual=str(decision.acceptable_level),
        )
    if expected_order_value is not None:
        if decision.order_value != expected_order_value:
            _add(
                issues,
                "order_value_mismatch",
                "order value differs from independent p x quantity",
                expected=str(expected_order_value),
                actual=str(decision.order_value),
            )
    if decision.approval_limit != snapshot.approval_limit:
        _add(issues, "approval_limit_mismatch", "decision approval limit differs from snapshot")
    if decision.policy_snapshot_id != snapshot.snapshot_id:
        _add(issues, "snapshot_id_mismatch", "decision used a different policy snapshot")
    if decision.policy_version != snapshot.policy_version:
        _add(issues, "policy_version_mismatch", "decision policy version differs from snapshot")
    if decision.quantity != extraction.total_quantity and extraction.is_total:
        _add(
            issues,
            "total_quantity_thread_mismatch",
            "total-offer quantity differs from the thread quantity",
            expected=str(extraction.total_quantity),
            actual=str(decision.quantity),
        )

    checks.extend(consistency_checks(decision, snapshot))
    provenance_complete = None
    evidence_status = None
    if evidence_package is not None:
        evidence_issues, evidence_checks, provenance_complete = verify_evidence_package(
            package=evidence_package,
            extraction=extraction,
            knowledge=knowledge,
            registry=registry,
        )
        issues.extend(evidence_issues)
        checks.extend(evidence_checks)
        evidence_status = "PASSED" if not evidence_issues else "FAILED"

    checks.extend(
        [
            VerificationCheck(
                name="snapshot_temporal_correctness",
                passed=not any(
                    issue.code
                    in {"snapshot_value_mismatch", "amendment_set_mismatch", "future_amendment_applied", "supplier_status_mismatch"}
                    for issue in issues
                ),
            ),
            VerificationCheck(
                name="extraction_grounded",
                passed=not any(
                    issue.code
                    in {
                        "price_candidate_mismatch",
                        "gst_detection_mismatch",
                        "billing_period_detection_mismatch",
                        "unit_factor_mismatch",
                        "offer_text_not_verbatim",
                        "previous_quote_candidate_mismatch",
                    }
                    for issue in issues
                ),
            ),
            VerificationCheck(
                name="normalization_independent",
                passed=not any(
                    issue.code in {"normalized_price_mismatch", "usability_mismatch"}
                    for issue in issues
                ),
            ),
            VerificationCheck(
                name="precedence_independent",
                passed=not any(
                    issue.code in {"final_label_mismatch", "applied_rule_mismatch", "injection_findings_mismatch", "banned_findings_mismatch"}
                    for issue in issues
                ),
            ),
            VerificationCheck(
                name="financial_arithmetic",
                passed=not any(
                    issue.code
                    in {"counter_price_mismatch", "acceptable_level_mismatch", "order_value_mismatch", "approval_limit_mismatch"}
                    for issue in issues
                ),
            ),
        ]
    )
    error_issues = [issue for issue in issues if issue.severity == "error"]
    status = VerificationStatus.PASSED
    if error_issues:
        status = VerificationStatus.FAILED
    elif corrected:
        status = VerificationStatus.PASSED_AFTER_CORRECTION
    return VerificationResult(
        thread_id=decision.thread_id,
        status=status,
        attempts=attempt,
        checks=tuple(checks),
        issues=tuple(issues),
        independent_normalized_price=independent_price,
        independent_acceptable_level=expected_level,
        independent_order_value=expected_order_value,
        independent_expected_label=expected_label,
        decision_label=decision.label,
        evidence_status=evidence_status,
        provenance_complete=provenance_complete,
        corrected=corrected,
    )
