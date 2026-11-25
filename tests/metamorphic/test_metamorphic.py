
from tests.helpers import evaluate_message, normalized_price


def labels_for(knowledge, messages, category="headsets", supplier="Monsoon Foods", date="2026-11-10", quantity=10, round_number=1):
    return [
        evaluate_message(
            knowledge,
            category=category,
            supplier=supplier,
            date=date,
            message=message,
            quantity=quantity,
            round_number=round_number,
        ).decision.label
        for message in messages
    ]


def test_irrelevant_text_does_not_change_decision(knowledge):
    base = "For the 10 headsets you requested, we can do ₹3,000 per headset."
    variants = [
        base,
        base + "\nOur AI-powered route planning keeps delivery times short.",
        base + "\nOur system integration team will handle setup at no extra cost.",
        base + "\nThe invoice will be pushed straight into your ERP system.",
        base + "\nOur assistant manager, Ravi, will coordinate delivery.",
        base + "\n" + ("Irrelevant background. " * 100),
    ]
    labels = labels_for(knowledge, variants)
    assert len(set(labels)) == 1
    assert labels[0] == "ACCEPT:ACCEPTABLE_PRICE:"


def test_price_increase_moves_through_decision_regions(knowledge):
    labels = labels_for(
        knowledge,
        [
            "For the 10 headsets you requested, we can do ₹3,000 per headset.",
            "For the 10 headsets you requested, we can do ₹4,000 per headset.",
        ],
    )
    assert labels == ["ACCEPT:ACCEPTABLE_PRICE:", "COUNTER:COUNTER_OFFER:3500"]
    final_label = labels_for(
        knowledge,
        ["For the 10 headsets you requested, we can do ₹5,000 per headset."],
        round_number=3,
    )[0]
    assert final_label == "REJECT:ABOVE_WALKAWAY:"


def test_date_crossing_amendment_changes_decision(knowledge):
    before = evaluate_message(
        knowledge,
        category="courier",
        supplier="Yamuna Traders",
        date="2026-09-21",
        message="For the 5 shipments you requested, we can do ₹205 per shipment.",
        quantity=5,
        round_number=4,
    ).decision.label
    after = evaluate_message(
        knowledge,
        category="courier",
        supplier="Yamuna Traders",
        date="2026-09-22",
        message="For the 5 shipments you requested, we can do ₹205 per shipment.",
        quantity=5,
        round_number=4,
    ).decision.label
    assert before == "REJECT:ABOVE_WALKAWAY:"
    assert after == "ACCEPT:ACCEPTABLE_PRICE:"


def test_date_crossing_approval_amendment_changes_decision(knowledge):
    before = evaluate_message(
        knowledge,
        category="headsets",
        supplier="Monsoon Foods",
        date="2026-11-08",
        message="For the 100 headsets you requested, we can do ₹3,201 per headset.",
        quantity=100,
        round_number=1,
    ).decision.label
    after = evaluate_message(
        knowledge,
        category="headsets",
        supplier="Monsoon Foods",
        date="2026-11-09",
        message="For the 100 headsets you requested, we can do ₹3,201 per headset.",
        quantity=100,
        round_number=1,
    ).decision.label
    assert before == "ACCEPT:ACCEPTABLE_PRICE:"
    assert after == "ESCALATE:OVER_APPROVAL_LIMIT:"


def test_lower_end_of_range_does_not_change_decision(knowledge):
    high_only = normalized_price(
        knowledge,
        category="monitors",
        supplier="Jupiter Devices",
        date="2026-11-10",
        message="For the 10 monitors you requested, we can do ₹14,600 per monitor.",
        quantity=10,
    )
    with_low = normalized_price(
        knowledge,
        category="monitors",
        supplier="Jupiter Devices",
        date="2026-11-10",
        message="For the 10 monitors you requested, we can do between ₹13,400 and ₹14,600 per monitor.",
        quantity=10,
    )
    assert high_only.normalized_price_inr_excl_gst == with_low.normalized_price_inr_excl_gst
