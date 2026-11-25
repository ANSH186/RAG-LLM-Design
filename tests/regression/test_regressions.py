from decimal import Decimal

from tests.helpers import normalized_price


def test_tier_threshold_is_inclusive(knowledge):
    at_threshold = normalized_price(
        knowledge,
        category="mobile-phones",
        supplier="Jupiter Devices",
        date="2026-10-01",
        message=(
            "For the 50 phones you requested, we can do INR 30,084 per phone for orders of up to "
            "50 phones, and ₹28,381 per phone for larger orders."
        ),
        quantity=50,
    )
    above_threshold = normalized_price(
        knowledge,
        category="mobile-phones",
        supplier="Jupiter Devices",
        date="2026-10-01",
        message=(
            "For the 51 phones you requested, we can do INR 30,084 per phone for orders of up to "
            "50 phones, and ₹28,381 per phone for larger orders."
        ),
        quantity=51,
    )
    assert at_threshold.normalized_price_inr_excl_gst == Decimal(30084)
    assert above_threshold.normalized_price_inr_excl_gst == Decimal(28381)


def test_volume_threshold_is_inclusive(knowledge):
    at_threshold = normalized_price(
        knowledge,
        category="catering",
        supplier="Ember Telecom",
        date="2026-10-01",
        message="For the 50 meals you requested, we can do ₹330 per meal, with a further 10% off for orders of 50 meals or more.",
        quantity=50,
    )
    below_threshold = normalized_price(
        knowledge,
        category="catering",
        supplier="Ember Telecom",
        date="2026-10-01",
        message="For the 49 meals you requested, we can do ₹330 per meal, with a further 10% off for orders of 50 meals or more.",
        quantity=49,
    )
    assert at_threshold.normalized_price_inr_excl_gst == Decimal("297.0")
    assert below_threshold.normalized_price_inr_excl_gst == Decimal(330)


def test_ream_conversion_regression(knowledge):
    result = normalized_price(
        knowledge,
        category="printer-paper",
        supplier="Indus Paper Mills",
        date="2026-10-01",
        message="For the 10 boxs you requested, we can do ₹257 per ream.",
        quantity=10,
    )
    assert result.normalized_price_inr_excl_gst == Decimal(2570)


def test_category_chunk_has_category_metadata(knowledge):
    from src.retrieval.chunking import build_chunks

    chunks = build_chunks(knowledge)
    category_chunks = [chunk for chunk in chunks if chunk.document_id == "laptops"]
    assert category_chunks
    assert all(chunk.category == "laptops" for chunk in category_chunks)


def test_previous_quote_not_acme_counter_regression(knowledge):
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
    assert result.normalized_price_inr_excl_gst == Decimal("3714.5")
