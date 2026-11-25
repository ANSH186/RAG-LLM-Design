"""Concession schedule L(r) with full Decimal precision."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation

from src.knowledge.policy_snapshot import PolicySnapshot
from src.knowledge.schema import SupplierStatus


def acceptable_level(snapshot: PolicySnapshot, round_number: int) -> Decimal:
    """L(r) = target + (walkaway - target) * ((r-1)/(R-1))**beta.

    Probation suppliers receive no concessions: L(r) equals target in every round.
    The returned level is not rounded; only a COUNTER price is rounded for output.
    """
    rounds = snapshot.negotiation_rounds
    if rounds < 2:
        raise ValueError("negotiation rounds must be at least 2")
    if not (1 <= round_number <= rounds):
        raise ValueError(f"round {round_number} outside 1..{rounds}")
    if snapshot.supplier_status == SupplierStatus.PROBATION:
        return snapshot.target_price
    if round_number == 1:
        return snapshot.target_price
    if round_number == rounds:
        return snapshot.walkaway_price
    ratio = Decimal(round_number - 1) / Decimal(rounds - 1)
    try:
        factor = ratio ** snapshot.concession_beta
    except InvalidOperation as exc:  # pragma: no cover - defensive
        raise ValueError(f"cannot evaluate concession exponent {snapshot.concession_beta}") from exc
    return snapshot.target_price + (snapshot.walkaway_price - snapshot.target_price) * factor


def is_final_round(snapshot: PolicySnapshot, round_number: int) -> bool:
    return round_number >= snapshot.negotiation_rounds
