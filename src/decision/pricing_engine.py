"""Pricing helpers used by the decision engine and independent verifier."""
from __future__ import annotations

from decimal import Decimal

from src.decision.concession import acceptable_level
from src.decision.precedence import DecisionAction, DecisionReason
from src.knowledge.policy_snapshot import PolicySnapshot
from src.utils.numbers import round_half_up


def order_value(normalized_price: Decimal, quantity: int) -> Decimal:
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    return normalized_price * Decimal(quantity)


def exceeds_approval_limit(normalized_price: Decimal, quantity: int, limit: Decimal) -> bool:
    return order_value(normalized_price, quantity) > limit


def price_decision(
    *, normalized_price: Decimal, snapshot: PolicySnapshot, round_number: int
) -> tuple[DecisionAction, DecisionReason, int | None, Decimal]:
    """Apply step 5 only. Returns action, reason, counter price, acceptable level."""
    level = acceptable_level(snapshot, round_number)
    if normalized_price <= level:
        return DecisionAction.ACCEPT, DecisionReason.ACCEPTABLE_PRICE, None, level
    if round_number >= snapshot.negotiation_rounds:
        return DecisionAction.REJECT, DecisionReason.ABOVE_WALKAWAY, None, level
    counter = round_half_up(level)
    return DecisionAction.COUNTER, DecisionReason.COUNTER_OFFER, counter, level


def decision_label(action: DecisionAction, reason: DecisionReason, counter_price: int | None) -> str:
    if action == DecisionAction.COUNTER:
        if counter_price is None:
            raise ValueError("counter decision requires a counter price")
        return f"{action.value}:{reason.value}:{counter_price}"
    return f"{action.value}:{reason.value}:"
