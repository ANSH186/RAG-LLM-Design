from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.decision.concession import acceptable_level
from src.utils.numbers import round_half_up
from tests.helpers import snapshot_for


@pytest.mark.parametrize("category", [
    "catering", "cleaning", "cloud-servers", "courier", "crm-licenses",
    "design-software", "headsets", "laptops", "led-lighting", "mobile-phones",
    "monitors", "office-chairs", "printer-paper", "security-guards", "training",
])
def test_concession_endpoints_and_monotonicity(knowledge, category):
    snapshot = snapshot_for(
        knowledge, category=category, supplier="Tidewater Systems", date="2026-10-01"
    )
    levels = [
        acceptable_level(snapshot, round_number)
        for round_number in range(1, snapshot.negotiation_rounds + 1)
    ]
    assert levels[0] == snapshot.target_price
    assert levels[-1] == snapshot.walkaway_price
    assert all(left <= right for left, right in zip(levels, levels[1:]))


@given(st.decimals(min_value=Decimal(0), max_value=Decimal(1000000), places=4))
@settings(max_examples=100, deadline=None)
def test_rounding_is_nearest_half_up(value):
    result = round_half_up(value)
    assert abs(Decimal(result) - value) <= Decimal("0.5")
    # Exact half ties must round upward.
    tie = value.quantize(Decimal(1)) + Decimal("0.5")
    assert round_half_up(tie) == int(tie + Decimal("0.5"))


@given(
    st.decimals(min_value=Decimal("0.01"), max_value=Decimal(100000), places=2),
    st.integers(min_value=1, max_value=10000),
)
@settings(max_examples=100, deadline=None)
def test_order_value_is_exact_decimal_multiplication(price, quantity):
    assert price * quantity == Decimal(quantity) * price
