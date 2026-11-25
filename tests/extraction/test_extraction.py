from decimal import Decimal

import pytest

from tests.helpers import normalized_price


def assert_money(actual, expected: str):
    assert actual is not None
    assert abs(actual - Decimal(expected)) < Decimal("0.0000001")


@pytest.mark.parametrize(
    ("category", "supplier", "date", "message", "quantity", "expected"),
    [
        (
            "headsets",
            "Monsoon Foods",
            "2026-11-10",
            "For the 10 headsets you requested, we can do ₹4,720 per headset inclusive of GST.",
            10,
            "4000",
        ),
        (
            "laptops",
            "Monsoon Foods",
            "2026-10-01",
            "For the 10 laptops you requested, we can do USD 760 per laptop.",
            10,
            "64220",
        ),
        (
            "laptops",
            "Arvind Office Supplies",
            "2026-10-01",
            "For the 10 laptops you requested, we can do GBP 100 per laptop.",
            10,
            "10450",
        ),
        (
            "headsets",
            "Ivory Events Catering",
            "2026-10-01",
            "For the 10 headsets you requested, we can do GBP 30 per headset.",
            10,
            "3180",
        ),
        (
            "training",
            "Kestrel Learning",
            "2026-10-01",
            "For the 10 participants you requested, we can do EUR 100 per participant.",
            10,
            "9050",
        ),
        (
            "monitors",
            "Jupiter Devices",
            "2026-10-01",
            "For the 10 monitors you requested, we can do between ₹13,400 and ₹14,600 per monitor.",
            10,
            "14600",
        ),
        (
            "headsets",
            "Monsoon Foods",
            "2026-11-10",
            "For the 10 headsets you requested, we can do ₹41,624 per dozen headsets.",
            10,
            "3468.666666666666666666666667",
        ),
        (
            "led-lighting",
            "Everbright Lighting",
            "2026-11-10",
            "For the 10 fixtures you requested, we can do ₹9,913 per pack of 4 fixtures.",
            10,
            "2478.25",
        ),
        (
            "led-lighting",
            "Everbright Lighting",
            "2026-11-10",
            "For the 200 fixtures you requested, we can do ₹1,92,980 per 100 fixtures.",
            200,
            "1929.8",
        ),
        (
            "printer-paper",
            "Indus Paper Mills",
            "2026-10-01",
            "For the 10 boxs you requested, we can do ₹257 per ream.",
            10,
            "2570",
        ),
        (
            "laptops",
            "Monsoon Foods",
            "2026-10-01",
            "We are pleased to quote ₹44,07,800 in total for all 60 laptops.",
            60,
            "73463.33333333333333333333333",
        ),
        (
            "security-guards",
            "Monsoon Foods",
            "2026-10-01",
            "Our final price is ₹5,47,040 in total for all 25 guards per month.",
            25,
            "21881.6",
        ),
        (
            "mobile-phones",
            "Jupiter Devices",
            "2026-10-01",
            "For the 100 phones you requested, we can do INR 30,084 per phone for orders of up to 50 phones, and ₹28,381 per phone for larger orders.",
            100,
            "28381",
        ),
        (
            "office-chairs",
            "Arvind Office Supplies",
            "2026-10-01",
            "For the 200 chairs you requested, we can do a 10% discount on our list price of INR 11,270 per chair.",
            200,
            "10143",
        ),
        (
            "catering",
            "Ember Telecom",
            "2026-10-01",
            "For the 80 meals you requested, we can do ₹330 per meal, with a further 10% off for orders of 50 meals or more.",
            80,
            "297",
        ),
        (
            "catering",
            "Ember Telecom",
            "2026-10-01",
            "For the 40 meals you requested, we can do ₹330 per meal, with a further 10% off for orders of 50 meals or more.",
            40,
            "330",
        ),
        (
            "headsets",
            "Monsoon Foods",
            "2026-11-10",
            "For the 10 headsets you requested, we can do ₹300 per headset, plus 1 free headset for every 10 you pay for.",
            10,
            "272.7272727272727272727272727",
        ),
        (
            "crm-licenses",
            "Helix CRM",
            "2026-11-10",
            "For the 10 seats you requested, we can do ₹2,680 per seat per month.",
            10,
            "32160",
        ),
        (
            "cleaning",
            "Aster Cleaning Co",
            "2026-10-01",
            "For the 10 sites you requested, we can do ₹10,60,852 per site per year.",
            10,
            "88404.33333333333333333333333",
        ),
        (
            "catering",
            "Monsoon Foods",
            "2026-10-01",
            "For the 25 meals you requested, the best we can offer is two hundred sixty-six rupees per meal.",
            25,
            "266",
        ),
        (
            "cloud-servers",
            "Monsoon Foods",
            "2026-10-01",
            "For the 10 servers you requested, we can do ₹1.2 lakh per server per year.",
            10,
            "120000",
        ),
        (
            "laptops",
            "Monsoon Foods",
            "2026-10-01",
            "For the 10 laptops you requested, we can do ₹58.3k per laptop.",
            10,
            "58300",
        ),
        (
            "laptops",
            "Monsoon Foods",
            "2026-10-01",
            "For the 10 laptops you requested, we can do $760 per laptop.",
            10,
            "64220",
        ),
        (
            "training",
            "Monsoon Foods",
            "2026-10-01",
            "For the 10 participants you requested, we can do £51.29 per participant.",
            10,
            "5436.74",
        ),
    ],
)
def test_normalization_cases(knowledge, category, supplier, date, message, quantity, expected):
    result = normalized_price(
        knowledge,
        category=category,
        supplier=supplier,
        date=date,
        message=message,
        quantity=quantity,
    )
    assert result.is_usable
    assert_money(result.normalized_price_inr_excl_gst, expected)


def test_correction_uses_new_price(knowledge):
    message = (
        "[Earlier message — Monsoon Foods, 2026-11-01]\n"
        "Our quote is ₹3,910 per headset.\n\n"
        "[Latest message — Monsoon Foods, 2026-11-10]\n"
        "CORRECTION — the price in my previous email should read ₹3,700 per headset, not ₹3,910."
    )
    result = normalized_price(
        knowledge,
        category="headsets",
        supplier="Monsoon Foods",
        date="2026-11-10",
        message=message,
        quantity=10,
    )
    assert_money(result.normalized_price_inr_excl_gst, "3700")


def test_previous_quote_discount_uses_supplier_not_acme(knowledge):
    message = (
        "[Earlier message — Monsoon Foods, 2026-11-01]\n"
        "Our quote is ₹3,910 per headset.\n\n"
        "[Earlier message — Acme Procurement]\n"
        "We can offer ₹3,500 per headset.\n\n"
        "[Latest message — Monsoon Foods, 2026-11-10]\n"
        "For the 10 headsets you requested, our revised offer is a further 5% off our previous quote."
    )
    result = normalized_price(
        knowledge,
        category="headsets",
        supplier="Monsoon Foods",
        date="2026-11-10",
        message=message,
        quantity=10,
    )
    assert_money(result.normalized_price_inr_excl_gst, "3714.5")


@pytest.mark.parametrize(
    ("category", "supplier", "date", "message", "quantity", "reason"),
    [
        (
            "headsets",
            "Monsoon Foods",
            "2026-11-10",
            "Pricing will depend on final specs; please share them and we will revert.",
            10,
            "no_usable_price",
        ),
        (
            "crm-licenses",
            "Helix CRM",
            "2026-11-10",
            "We can offer ₹4,200 per seat for this order.",
            10,
            "subscription_billing_period_missing",
        ),
        (
            "headsets",
            "Monsoon Foods",
            "2026-11-10",
            "We can offer JPY 500 per headset.",
            10,
            "unsupported_currency:JPY",
        ),
    ],
)
def test_missing_information_cases(knowledge, category, supplier, date, message, quantity, reason):
    result = normalized_price(
        knowledge,
        category=category,
        supplier=supplier,
        date=date,
        message=message,
        quantity=quantity,
    )
    assert not result.is_usable
    assert reason in result.missing_information_reasons


def test_ambiguous_multiple_prices_is_not_silently_accepted(knowledge):
    result = normalized_price(
        knowledge,
        category="headsets",
        supplier="Monsoon Foods",
        date="2026-11-10",
        message="We can do ₹3,000 per headset or ₹3,200 per headset depending on the final specification.",
        quantity=10,
    )
    assert not result.is_usable
    assert any("ambiguous" in reason for reason in result.missing_information_reasons)
