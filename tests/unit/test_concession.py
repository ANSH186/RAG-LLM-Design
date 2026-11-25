from decimal import Decimal

import pytest

from src.decision.concession import acceptable_level
from src.decision.precedence import DecisionAction, DecisionReason
from src.decision.pricing_engine import price_decision
from tests.helpers import snapshot_for


def test_concession_endpoints_and_midpoint(knowledge):
    snap = snapshot_for(
        knowledge, category="catering", supplier="Monsoon Foods", date="2026-10-01"
    )
    assert acceptable_level(snap, 1) == Decimal(270)
    assert acceptable_level(snap, 2) == Decimal(305)
    assert acceptable_level(snap, 3) == Decimal(340)


def test_concession_full_precision(knowledge):
    snap = snapshot_for(knowledge, category="laptops", supplier="Monsoon Foods", date="2026-10-01")
    level = acceptable_level(snap, 2)
    assert level == Decimal("62888.88888888888888888888889")
    action, reason, counter, returned_level = price_decision(
        normalized_price=Decimal(71900), snapshot=snap, round_number=2
    )
    assert returned_level == level
    assert action == DecisionAction.COUNTER
    assert reason == DecisionReason.COUNTER_OFFER
    assert counter == 62889


def test_probation_no_concessions(knowledge):
    snap = snapshot_for(
        knowledge, category="security-guards", supplier="Banyan Guarding", date="2026-10-01"
    )
    for round_number in range(1, snap.negotiation_rounds + 1):
        assert acceptable_level(snap, round_number) == Decimal(21000)


def test_final_round_walkaway(knowledge):
    snap = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    assert acceptable_level(snap, 3) == Decimal(4500)
    action, reason, counter, _ = price_decision(
        normalized_price=Decimal("4500.01"), snapshot=snap, round_number=3
    )
    assert action == DecisionAction.REJECT
    assert reason == DecisionReason.ABOVE_WALKAWAY
    assert counter is None


def test_invalid_round_rejected(knowledge):
    snap = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    with pytest.raises(ValueError):
        acceptable_level(snap, 0)
    with pytest.raises(ValueError):
        acceptable_level(snap, snap.negotiation_rounds + 1)
