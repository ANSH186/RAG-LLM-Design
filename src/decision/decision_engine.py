"""The deterministic Acme procurement decision engine.

The LLM never determines the final action. This module applies the official
decision procedure in order, using the resolved policy snapshot and Decimal money.
"""
from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from src.decision.concession import acceptable_level
from src.decision.precedence import (
    RULE_ORDER,
    DecisionAction,
    DecisionReason,
    RuleName,
)
from src.decision.pricing_engine import (
    decision_label,
    exceeds_approval_limit,
    order_value,
    price_decision,
)
from src.decision.security_rules import SecurityFinding, scan_security
from src.extraction.extraction_schema import NormalizedOffer, OfferExtraction
from src.knowledge.policy_snapshot import PolicySnapshot
from src.knowledge.schema import SupplierStatus
from src.utils.hashing import sha256_json


class AppliedRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: RuleName
    outcome: str
    detail: str


class DecisionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    thread_id: str
    decision_id: str
    action: DecisionAction
    reason: DecisionReason
    label: str
    counter_price: int | None = None
    applied_rule: RuleName
    precedence_order: tuple[RuleName, ...] = RULE_ORDER
    policy_snapshot_id: str
    policy_version: str
    supplier_status: SupplierStatus
    round_number: int
    quantity: int
    category: str
    supplier: str
    normalized_price: Decimal | None = None
    acceptable_level: Decimal | None = None
    order_value: Decimal | None = None
    approval_limit: Decimal | None = None
    normalized_offer: NormalizedOffer
    extraction: OfferExtraction
    injection_findings: tuple[SecurityFinding, ...] = ()
    banned_term_findings: tuple[SecurityFinding, ...] = ()
    applied_rules: tuple[AppliedRule, ...] = ()
    calculations: dict[str, str] = Field(default_factory=dict)
    explanation: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def audit_summary(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "label": self.label,
            "action": self.action.value,
            "reason": self.reason.value,
            "counter_price": self.counter_price,
            "normalized_price_inr_excl_gst": str(self.normalized_price) if self.normalized_price is not None else None,
            "acceptable_level": str(self.acceptable_level) if self.acceptable_level is not None else None,
            "order_value": str(self.order_value) if self.order_value is not None else None,
            "approval_limit": str(self.approval_limit) if self.approval_limit is not None else None,
            "policy_snapshot_id": self.policy_snapshot_id,
            "policy_version": self.policy_version,
            "applied_rule": self.applied_rule.value,
        }


def decide(
    *,
    thread_id: str,
    category: str,
    supplier: str,
    round_number: int,
    quantity: int,
    extraction: OfferExtraction,
    normalized: NormalizedOffer,
    snapshot: PolicySnapshot,
    injection_findings: list[SecurityFinding] | None = None,
    banned_term_findings: list[SecurityFinding] | None = None,
) -> DecisionResult:
    if snapshot.thread_id != thread_id:
        raise ValueError("policy snapshot thread mismatch")
    if injection_findings is None or banned_term_findings is None:
        scanned_injections, scanned_banned = scan_security(extraction.source_message)
        injection_findings = injection_findings if injection_findings is not None else scanned_injections
        banned_term_findings = banned_term_findings if banned_term_findings is not None else scanned_banned

    applied: list[AppliedRule] = []
    explanation: list[str] = []
    calculations: dict[str, str] = {
        "quantity": str(quantity),
        "round": str(round_number),
        "category": category,
        "supplier": supplier,
    }
    warnings: list[str] = list(normalized.warnings)

    def finish(
        action: DecisionAction,
        reason: DecisionReason,
        rule: RuleName,
        counter_price: int | None = None,
        *,
        acceptable_level_value: Decimal | None = None,
        order_value_value: Decimal | None = None,
    ) -> DecisionResult:
        label = decision_label(action, reason, counter_price)
        applied.append(AppliedRule(rule=rule, outcome=f"{action.value}:{reason.value}", detail=label))
        payload = {
            "thread": thread_id,
            "snapshot": snapshot.snapshot_id,
            "label": label,
            "normalized": str(normalized.normalized_price_inr_excl_gst),
            "qty": quantity,
        }
        return DecisionResult(
            thread_id=thread_id,
            decision_id="dec-" + sha256_json(payload)[:16],
            action=action,
            reason=reason,
            label=label,
            counter_price=counter_price,
            applied_rule=rule,
            policy_snapshot_id=snapshot.snapshot_id,
            policy_version=snapshot.policy_version,
            supplier_status=snapshot.supplier_status,
            round_number=round_number,
            quantity=quantity,
            category=category,
            supplier=supplier,
            normalized_price=normalized.normalized_price_inr_excl_gst,
            acceptable_level=acceptable_level_value,
            order_value=order_value_value,
            approval_limit=snapshot.approval_limit,
            normalized_offer=normalized,
            extraction=extraction,
            injection_findings=tuple(injection_findings or ()),
            banned_term_findings=tuple(banned_term_findings or ()),
            applied_rules=tuple(applied),
            calculations=calculations,
            explanation=tuple(explanation),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    # Rule 1: missing information always fires first.
    if not normalized.is_usable:
        reasons = ", ".join(normalized.missing_information_reasons) or "no usable price"
        explanation.append(f"Latest supplier offer is not usable: {reasons}.")
        calculations["missing_information"] = reasons
        return finish(DecisionAction.ESCALATE, DecisionReason.MISSING_INFO, RuleName.MISSING_INFORMATION)

    # Rule 2: blacklisted supplier.
    if snapshot.supplier_status == SupplierStatus.BLACKLISTED:
        explanation.append(
            f"Supplier {snapshot.supplier_name} is BLACKLISTED in the official supplier contract."
        )
        return finish(DecisionAction.REJECT, DecisionReason.BLACKLISTED_SUPPLIER, RuleName.BLACKLISTED_SUPPLIER)

    # Rule 3: prompt injection anywhere in the thread.
    if injection_findings:
        sample = injection_findings[0].matched_text
        explanation.append(f"Prompt-injection language detected in the thread: {sample}")
        calculations["injection_findings"] = str(len(injection_findings))
        return finish(DecisionAction.ESCALATE, DecisionReason.PROMPT_INJECTION, RuleName.PROMPT_INJECTION)

    # Rule 4: banned contractual terms in the latest supplier message.
    if banned_term_findings:
        sample = banned_term_findings[0]
        explanation.append(f"Banned contractual term detected: {sample.details} ({sample.matched_text})")
        calculations["banned_term_findings"] = str(len(banned_term_findings))
        return finish(DecisionAction.ESCALATE, DecisionReason.BANNED_TERM, RuleName.BANNED_TERM)

    # Rule 5: price decision.
    if not (1 <= round_number <= snapshot.negotiation_rounds):
        explanation.append(
            f"Thread round {round_number} is outside the official range 1..{snapshot.negotiation_rounds}."
        )
        calculations["invalid_round"] = str(round_number)
        return finish(DecisionAction.ESCALATE, DecisionReason.MISSING_INFO, RuleName.MISSING_INFORMATION)

    normalized_price = normalized.normalized_price_inr_excl_gst
    if normalized_price is None:  # defensive; is_usable guarantees this
        explanation.append("Normalized price unexpectedly absent.")
        return finish(DecisionAction.ESCALATE, DecisionReason.MISSING_INFO, RuleName.MISSING_INFORMATION)

    level = acceptable_level(snapshot, round_number)
    calculations["normalized_price_inr_excl_gst"] = str(normalized_price)
    calculations["acceptable_level_L_r"] = str(level)
    calculations["target_price"] = str(snapshot.target_price)
    calculations["walkaway_price"] = str(snapshot.walkaway_price)
    calculations["concession_exponent"] = str(snapshot.concession_beta)
    calculations["negotiation_rounds"] = str(snapshot.negotiation_rounds)
    calculations["supplier_status"] = snapshot.supplier_status.value
    applied.append(
        AppliedRule(
            rule=RuleName.PRICE_DECISION,
            outcome="evaluated",
            detail=f"normalized {normalized_price} vs L({round_number})={level}",
        )
    )
    if normalized_price <= level:
        explanation.append(
            f"Normalized price {normalized_price} INR is at or below the round-{round_number} "
            f"acceptable level {level} INR."
        )
        # Rule 6: approval limit.
        total = order_value(normalized_price, quantity)
        calculations["order_value"] = str(total)
        calculations["approval_limit"] = str(snapshot.approval_limit)
        applied.append(
            AppliedRule(
                rule=RuleName.APPROVAL_LIMIT,
                outcome="evaluated",
                detail=f"order value {total} vs approval limit {snapshot.approval_limit}",
            )
        )
        if exceeds_approval_limit(normalized_price, quantity, snapshot.approval_limit):
            explanation.append(
                f"Order value {total} INR exceeds the approval limit {snapshot.approval_limit} INR."
            )
            return finish(
                DecisionAction.ESCALATE,
                DecisionReason.OVER_APPROVAL_LIMIT,
                RuleName.APPROVAL_LIMIT,
                acceptable_level_value=level,
                order_value_value=total,
            )
        explanation.append(
            f"Order value {total} INR is within the approval limit {snapshot.approval_limit} INR."
        )
        return finish(
            DecisionAction.ACCEPT,
            DecisionReason.ACCEPTABLE_PRICE,
            RuleName.PRICE_DECISION,
            acceptable_level_value=level,
            order_value_value=total,
        )

    action, reason, counter_price, level = price_decision(
        normalized_price=normalized_price, snapshot=snapshot, round_number=round_number
    )
    calculations["price_decision"] = f"{action.value}:{reason.value}"
    if action == DecisionAction.REJECT:
        explanation.append(
            f"Normalized price {normalized_price} INR is above the final-round walk-away level {level} INR."
        )
    else:
        explanation.append(
            f"Normalized price {normalized_price} INR is above the round-{round_number} "
            f"acceptable level {level} INR; counter at the level rounded to the nearest rupee."
        )
        calculations["counter_price"] = str(counter_price)
    return finish(
        action,
        reason,
        RuleName.PRICE_DECISION,
        counter_price,
        acceptable_level_value=level,
    )
