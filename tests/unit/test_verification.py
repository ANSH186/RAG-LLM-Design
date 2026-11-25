from decimal import Decimal

from tests.helpers import evaluate_message, evaluate_with_evidence, snapshot_for


def test_all_labelled_samples_pass_verification(
    knowledge, retriever, registry, context_builder, sample_threads
):
    failures = []
    for _, row in sample_threads.iterrows():
        evaluated, verification = evaluate_with_evidence(
            knowledge,
            retriever,
            registry,
            context_builder,
            category=row.category,
            supplier=row.supplier,
            date=row.date,
            message=row.supplier_message,
            quantity=int(row.quantity),
            round_number=int(row["round"]),
            thread_id=row.thread_id,
        )
        if not verification.passed:
            failures.append((row.thread_id, [issue.code for issue in verification.error_issues]))
    assert failures == []


def test_tampered_normalization_is_detected(knowledge, retriever, registry, context_builder, sample_threads):
    row = sample_threads[sample_threads.thread_id == "T00005"].iloc[0]
    evaluated, verification = evaluate_with_evidence(
        knowledge, retriever, registry, context_builder,
        category=row.category, supplier=row.supplier, date=row.date,
        message=row.supplier_message, quantity=int(row.quantity),
        round_number=int(row["round"]), thread_id=row.thread_id,
    )
    assert verification.passed
    tampered = evaluated.normalized.model_copy(
        update={"normalized_price_inr_excl_gst": Decimal(1)}
    )
    from src.verification.decision_verifier import verify_decision

    result = verify_decision(
        decision=evaluated.decision,
        extraction=evaluated.extraction,
        normalized=tampered,
        snapshot=evaluated.snapshot,
        knowledge=knowledge,
    )
    assert not result.passed
    assert any(issue.code == "normalized_price_mismatch" for issue in result.error_issues)


def test_tampered_label_is_detected(knowledge, retriever, registry, context_builder, sample_threads):
    row = sample_threads[sample_threads.thread_id == "T00010"].iloc[0]
    evaluated, verification = evaluate_with_evidence(
        knowledge, retriever, registry, context_builder,
        category=row.category, supplier=row.supplier, date=row.date,
        message=row.supplier_message, quantity=int(row.quantity),
        round_number=int(row["round"]), thread_id=row.thread_id,
    )
    assert evaluated.decision.label == "COUNTER:COUNTER_OFFER:85000"
    tampered = evaluated.decision.model_copy(update={"label": "ACCEPT:ACCEPTABLE_PRICE:"})
    from src.verification.decision_verifier import verify_decision

    result = verify_decision(
        decision=tampered,
        extraction=evaluated.extraction,
        normalized=evaluated.normalized,
        snapshot=evaluated.snapshot,
        knowledge=knowledge,
    )
    assert not result.passed
    assert any(issue.code == "final_label_mismatch" for issue in result.error_issues)


def test_tampered_snapshot_value_is_detected(knowledge):
    snapshot = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    tampered = snapshot.model_copy(update={"walkaway_price": Decimal(9999)})
    evaluated = evaluate_message(
        knowledge,
        category="headsets", supplier="Monsoon Foods", date="2026-11-10",
        message="For the 10 headsets you requested, we can do ₹3,000 per headset.",
        quantity=10, round_number=1,
    )
    from src.verification.decision_verifier import verify_decision

    result = verify_decision(
        decision=evaluated.decision,
        extraction=evaluated.extraction,
        normalized=evaluated.normalized,
        snapshot=tampered,
        knowledge=knowledge,
    )
    assert not result.passed
    assert any(issue.code == "snapshot_value_mismatch" for issue in result.error_issues)


def test_future_amendment_in_snapshot_is_detected(knowledge):
    snapshot = snapshot_for(knowledge, category="led-lighting", supplier="Everbright Lighting", date="2026-12-03")
    tampered = snapshot.model_copy(
        update={
            "walkaway_price": Decimal(2360),
            "amendments_applied": ("AMD-08", "AMD-07"),
        }
    )
    evaluated = evaluate_message(
        knowledge,
        category="led-lighting", supplier="Everbright Lighting", date="2026-12-03",
        message="For the 10 fixtures you requested, we can do ₹2,100 per fixture.",
        quantity=10, round_number=1,
    )
    from src.verification.decision_verifier import verify_decision

    result = verify_decision(
        decision=evaluated.decision,
        extraction=evaluated.extraction,
        normalized=evaluated.normalized,
        snapshot=tampered,
        knowledge=knowledge,
    )
    assert not result.passed
    assert any(issue.code == "future_amendment_applied" for issue in result.error_issues)


def test_counter_rounding_tamper_is_detected(knowledge):
    evaluated = evaluate_message(
        knowledge,
        category="laptops", supplier="Monsoon Foods", date="2026-10-01",
        message="For the 50 laptops you requested, we can do ₹71,900 per laptop.",
        quantity=50, round_number=2,
    )
    assert evaluated.decision.counter_price == 62889
    tampered = evaluated.decision.model_copy(update={"counter_price": 62888, "label": "COUNTER:COUNTER_OFFER:62888"})
    from src.verification.decision_verifier import verify_decision

    result = verify_decision(
        decision=tampered,
        extraction=evaluated.extraction,
        normalized=evaluated.normalized,
        snapshot=evaluated.snapshot,
        knowledge=knowledge,
    )
    assert not result.passed
    assert any(issue.code == "counter_price_mismatch" for issue in result.error_issues)
