"""Independent consistency checks for decision labels and financial invariants."""
from __future__ import annotations

import re

from src.decision.precedence import ACTION_FOR_REASON, DecisionAction, DecisionReason
from src.knowledge.policy_snapshot import PolicySnapshot
from src.knowledge.schema import SupplierStatus
from src.verification.schemas import VerificationCheck, VerificationIssue

LABEL_PATTERNS = {
    "ACCEPT:ACCEPTABLE_PRICE:": (DecisionAction.ACCEPT, DecisionReason.ACCEPTABLE_PRICE),
    "REJECT:ABOVE_WALKAWAY:": (DecisionAction.REJECT, DecisionReason.ABOVE_WALKAWAY),
    "REJECT:BLACKLISTED_SUPPLIER:": (DecisionAction.REJECT, DecisionReason.BLACKLISTED_SUPPLIER),
    "ESCALATE:MISSING_INFO:": (DecisionAction.ESCALATE, DecisionReason.MISSING_INFO),
    "ESCALATE:PROMPT_INJECTION:": (DecisionAction.ESCALATE, DecisionReason.PROMPT_INJECTION),
    "ESCALATE:BANNED_TERM:": (DecisionAction.ESCALATE, DecisionReason.BANNED_TERM),
    "ESCALATE:OVER_APPROVAL_LIMIT:": (DecisionAction.ESCALATE, DecisionReason.OVER_APPROVAL_LIMIT),
}
COUNTER_RE = re.compile(r"^COUNTER:COUNTER_OFFER:(\d+)$")


def check_label_shape(decision) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    label = decision.label
    if decision.action == DecisionAction.COUNTER:
        match = COUNTER_RE.match(label)
        if not match:
            issues.append(
                VerificationIssue(
                    code="invalid_counter_label",
                    detail="COUNTER label must be COUNTER:COUNTER_OFFER:<whole rupees>",
                    actual=label,
                )
            )
        elif decision.counter_price is None or int(match.group(1)) != decision.counter_price:
            issues.append(
                VerificationIssue(
                    code="counter_price_label_mismatch",
                    detail="counter price does not match the label",
                    expected=str(decision.counter_price),
                    actual=match.group(1),
                )
            )
        if decision.counter_price is None or decision.counter_price <= 0:
            issues.append(
                VerificationIssue(
                    code="counter_price_not_positive",
                    detail="counter price must be a positive whole rupee amount",
                    actual=str(decision.counter_price),
                )
            )
    else:
        expected = LABEL_PATTERNS.get(label)
        if expected is None:
            issues.append(
                VerificationIssue(
                    code="unknown_label",
                    detail="label is not one of the official output labels",
                    actual=label,
                )
            )
        elif expected != (decision.action, decision.reason):
            issues.append(
                VerificationIssue(
                    code="action_reason_label_mismatch",
                    detail="label action/reason does not match decision fields",
                    actual=f"{decision.action.value}:{decision.reason.value}",
                )
            )
        if decision.counter_price is not None:
            issues.append(
                VerificationIssue(
                    code="unexpected_counter_price",
                    detail="non-COUNTER decision carries a counter price",
                    actual=str(decision.counter_price),
                )
            )
    if ACTION_FOR_REASON.get(decision.reason) != decision.action:
        issues.append(
            VerificationIssue(
                code="action_reason_invalid",
                detail="action is not the official action for this reason",
                expected=ACTION_FOR_REASON.get(decision.reason, DecisionAction.ACCEPT).value,
                actual=decision.action.value,
            )
        )
    return issues


def check_snapshot_consistency(decision, snapshot: PolicySnapshot) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    if snapshot.thread_id != decision.thread_id:
        issues.append(
            VerificationIssue(
                code="snapshot_thread_mismatch",
                detail="policy snapshot belongs to another thread",
            )
        )
    if not (1 <= decision.round_number <= snapshot.negotiation_rounds):
        issues.append(
            VerificationIssue(
                code="round_out_of_range",
                detail="thread round is outside the official range",
                expected=f"1..{snapshot.negotiation_rounds}",
                actual=str(decision.round_number),
            )
        )
    if snapshot.policy_version.split(":")[0] != snapshot.policy_version.split(":")[0].lower():
        issues.append(VerificationIssue(code="policy_version_malformed", detail="policy version malformed"))
    if snapshot.negotiation_date.isoformat() not in snapshot.policy_version:
        issues.append(
            VerificationIssue(
                code="policy_version_date_mismatch",
                detail="policy version does not contain the negotiation date",
                actual=snapshot.policy_version,
            )
        )
    if snapshot.supplier_status == SupplierStatus.BLACKLISTED and decision.reason != DecisionReason.BLACKLISTED_SUPPLIER:
        # Only the earlier MISSING_INFORMATION rule may pre-empt blacklist status.
        if decision.reason != DecisionReason.MISSING_INFO:
            issues.append(
                VerificationIssue(
                    code="blacklist_not_applied",
                    detail="blacklisted supplier did not trigger the blacklist rule",
                    actual=decision.label,
                )
            )
    return issues


def check_financial_invariants(decision) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    normalized = decision.normalized_price
    if normalized is not None and normalized < 0:
        issues.append(
            VerificationIssue(
                code="negative_normalized_price",
                detail="normalized price is negative",
                actual=str(normalized),
            )
        )
    # Exact order-value identity when an order value is present.
    if decision.order_value is not None and decision.normalized_price is not None:
        # quantity is not duplicated on the decision, so recompute from extraction/thread elsewhere.
        if decision.approval_limit is not None and decision.reason == DecisionReason.OVER_APPROVAL_LIMIT:
            if decision.order_value <= decision.approval_limit:
                issues.append(
                    VerificationIssue(
                        code="over_limit_without_excess",
                        detail="OVER_APPROVAL_LIMIT raised without order value above limit",
                    )
                )
        if decision.reason == DecisionReason.ACCEPTABLE_PRICE and decision.approval_limit is not None:
            if decision.order_value > decision.approval_limit:
                issues.append(
                    VerificationIssue(
                        code="accept_above_limit",
                        detail="accepted decision is above the approval limit",
                    )
                )
    if decision.acceptable_level is not None and decision.normalized_price is not None:
        if decision.reason == DecisionReason.ACCEPTABLE_PRICE and decision.normalized_price > decision.acceptable_level:
            issues.append(
                VerificationIssue(
                    code="accept_above_level",
                    detail="accepted decision is above the acceptable level",
                )
            )
        if decision.reason == DecisionReason.ABOVE_WALKAWAY and decision.normalized_price <= decision.acceptable_level:
            issues.append(
                VerificationIssue(
                    code="reject_below_level",
                    detail="rejected decision is not above the acceptable level",
                )
            )
        if decision.reason == DecisionReason.COUNTER_OFFER:
            from src.utils.numbers import round_half_up

            if decision.counter_price != round_half_up(decision.acceptable_level):
                issues.append(
                    VerificationIssue(
                        code="counter_rounding_mismatch",
                        detail="counter price is not the level rounded half-up to whole rupees",
                        expected=str(round_half_up(decision.acceptable_level)),
                        actual=str(decision.counter_price),
                    )
                )
    return issues


def consistency_checks(decision, snapshot: PolicySnapshot) -> tuple[VerificationCheck, ...]:
    label_issues = check_label_shape(decision)
    snapshot_issues = check_snapshot_consistency(decision, snapshot)
    financial_issues = check_financial_invariants(decision)
    return (
        VerificationCheck(name="label_schema", passed=not label_issues, detail="; ".join(i.code for i in label_issues) or None),
        VerificationCheck(
            name="snapshot_consistency",
            passed=not snapshot_issues,
            detail="; ".join(i.code for i in snapshot_issues) or None,
        ),
        VerificationCheck(
            name="financial_invariants",
            passed=not financial_issues,
            detail="; ".join(i.code for i in financial_issues) or None,
        ),
    )
