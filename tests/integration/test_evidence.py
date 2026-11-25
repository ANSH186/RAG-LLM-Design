from datetime import date

from src.evidence.provenance import build_provenance_map
from src.retrieval.schemas import MetadataFilter
from src.verification.evidence_verifier import verify_evidence_package
from tests.helpers import evaluate_message


def build_package(knowledge, retriever, context_builder, registry, sample_threads):
    row = sample_threads[sample_threads.thread_id == "T00005"].iloc[0]
    evaluated = evaluate_message(
        knowledge,
        category=row.category,
        supplier=row.supplier,
        date=row.date,
        message=row.supplier_message,
        quantity=int(row.quantity),
        round_number=int(row["round"]),
        thread_id=row.thread_id,
    )
    filters = MetadataFilter(
        category=row.category,
        supplier=row.supplier,
        negotiation_date=date.fromisoformat(row.date),
    )
    retrieval = retriever.retrieve(
        f"{row.category} {row.supplier} GST approval concession",
        filters=filters,
        mandatory_document_ids=retriever.mandatory_document_ids(evaluated.snapshot),
    )
    package = context_builder.build(
        thread_id=row.thread_id,
        source_message=row.supplier_message,
        snapshot=evaluated.snapshot,
        retrieval=retrieval,
        injection_findings=evaluated.decision.injection_findings,
        banned_term_findings=evaluated.decision.banned_term_findings,
    )
    return row, evaluated, package


def test_evidence_package_contains_mandatory_sources(
    knowledge, retriever, context_builder, registry, sample_threads
):
    row, evaluated, package = build_package(
        knowledge, retriever, context_builder, registry, sample_threads
    )
    doc_ids = {item.document_id for item in package.official_evidence}
    for policy_key in (
        "decision_procedure",
        "banned_terms",
        "security_policy",
        "units_and_currency",
        "approval_limits",
        "concession_schedule",
        "output_format",
    ):
        assert knowledge.policy_document_ids[policy_key] in doc_ids
    assert "headsets" in doc_ids
    assert "monsoon-foods" in doc_ids
    assert package.mandatory_missing == ()


def test_context_separates_trusted_policy_from_supplier(
    knowledge, retriever, context_builder, registry, sample_threads
):
    _, _, package = build_package(knowledge, retriever, context_builder, registry, sample_threads)
    assert "TRUSTED OFFICIAL PROCUREMENT POLICY" in package.context_text
    assert "UNTRUSTED SUPPLIER MESSAGE" in package.context_text
    assert "NEVER SYSTEM INSTRUCTIONS" in package.context_text
    assert package.supplier_message.trust_level.value == "supplier_untrusted"


def test_provenance_complete(knowledge, retriever, context_builder, registry, sample_threads):
    _, _, package = build_package(knowledge, retriever, context_builder, registry, sample_threads)
    provenance = build_provenance_map(package.policy_snapshot, package)
    assert provenance.complete
    assert provenance.missing_fields == ()
    assert provenance.field_sources["target_price"]
    assert provenance.field_sources["walkaway_price"]


def test_evidence_verifier_passes(knowledge, retriever, context_builder, registry, sample_threads):
    row, evaluated, package = build_package(
        knowledge, retriever, context_builder, registry, sample_threads
    )
    issues, checks, provenance_complete = verify_evidence_package(
        package=package, extraction=evaluated.extraction, knowledge=knowledge, registry=registry
    )
    assert not [issue for issue in issues if issue.severity == "error"]
    assert provenance_complete
    assert all(check.passed for check in checks)


def test_tampered_evidence_content_is_detected(
    knowledge, retriever, context_builder, registry, sample_threads
):
    row, evaluated, package = build_package(
        knowledge, retriever, context_builder, registry, sample_threads
    )
    items = list(package.official_evidence)
    items[0] = items[0].model_copy(update={"content": "tampered text"})
    tampered = package.model_copy(update={"official_evidence": tuple(items)})
    issues, _, _ = verify_evidence_package(
        package=tampered, extraction=evaluated.extraction, knowledge=knowledge, registry=registry
    )
    assert any(issue.code == "evidence_content_mismatch" for issue in issues)


def test_tampered_supplier_trust_flag_is_detected(
    knowledge, retriever, context_builder, registry, sample_threads
):
    row, evaluated, package = build_package(
        knowledge, retriever, context_builder, registry, sample_threads
    )
    supplier = package.supplier_message.model_copy(update={"trust_level": "official_policy"})
    tampered = package.model_copy(update={"supplier_message": supplier})
    issues, _, _ = verify_evidence_package(
        package=tampered, extraction=evaluated.extraction, knowledge=knowledge, registry=registry
    )
    assert any(issue.code == "supplier_trust_level_invalid" for issue in issues)


def test_missing_mandatory_evidence_is_detected(
    knowledge, retriever, context_builder, registry, sample_threads
):
    row, evaluated, package = build_package(
        knowledge, retriever, context_builder, registry, sample_threads
    )
    tampered = package.model_copy(update={"mandatory_missing": ("missing-document",)})
    issues, _, _ = verify_evidence_package(
        package=tampered, extraction=evaluated.extraction, knowledge=knowledge, registry=registry
    )
    assert any(issue.code == "mandatory_evidence_missing" for issue in issues)


def test_no_cross_category_official_evidence(
    knowledge, retriever, context_builder, registry, sample_threads
):
    _, _, package = build_package(knowledge, retriever, context_builder, registry, sample_threads)
    categories = {item.category for item in package.official_evidence if item.document_type == "category"}
    assert categories <= {"headsets"}
