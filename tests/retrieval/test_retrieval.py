from datetime import date

from src.retrieval.rank_fusion import reciprocal_rank_fusion
from src.retrieval.schemas import MetadataFilter


def test_bm25_exact_banned_terms(retriever):
    result = retriever.retrieve(
        "automatic renewal exclusivity unlimited liability advance payment",
        filters=MetadataFilter(category="headsets", supplier="Monsoon Foods", negotiation_date=date(2026, 11, 10)),
    )
    assert result.hits
    assert result.hits[0].document_id == "banned-terms"
    assert result.hits[0].bm25_score is not None


def test_dense_retrieval_contributes(retriever):
    result = retriever.retrieve(
        "contract that keeps rolling forward without cancellation",
        filters=MetadataFilter(category="headsets", supplier="Monsoon Foods", negotiation_date=date(2026, 11, 10)),
    )
    assert "dense" in result.methods
    assert any(hit.dense_score is not None for hit in result.hits)


def test_temporal_filter_excludes_future_amendments(retriever):
    before = retriever.retrieve(
        "LED lighting walk-away price amendment",
        filters=MetadataFilter(category="led-lighting", supplier="Everbright Lighting", negotiation_date=date(2026, 12, 3)),
    )
    after = retriever.retrieve(
        "LED lighting walk-away price amendment",
        filters=MetadataFilter(category="led-lighting", supplier="Everbright Lighting", negotiation_date=date(2026, 12, 10)),
    )
    assert all(hit.document_id not in {"AMD-07", "AMD-08"} for hit in before.hits)
    assert any(hit.document_id in {"AMD-07", "AMD-08"} for hit in after.hits)


def test_category_filter_blocks_other_categories(retriever):
    result = retriever.retrieve(
        "printer paper target price box ream",
        filters=MetadataFilter(category="headsets", supplier="Monsoon Foods", negotiation_date=date(2026, 11, 10)),
        top_k=20,
    )
    assert all(hit.document_id not in {"printer-paper", "laptops", "led-lighting"} for hit in result.hits)


def test_supplier_filter_blocks_other_suppliers(retriever):
    result = retriever.retrieve(
        "supplier contract price cap exchange rate",
        filters=MetadataFilter(category="headsets", supplier="Monsoon Foods", negotiation_date=date(2026, 11, 10)),
        top_k=20,
    )
    supplier_docs = [hit.document_id for hit in result.hits if hit.document_type == "supplier"]
    assert all(doc == "monsoon-foods" for doc in supplier_docs)


def test_duplicate_removal_and_source_diversity(retriever):
    result = retriever.retrieve(
        "decision missing information blacklisted prompt injection banned terms price approval limit",
        filters=MetadataFilter(category="laptops", supplier="Monsoon Foods", negotiation_date=date(2026, 11, 10)),
        top_k=20,
    )
    chunk_ids = [hit.chunk_id for hit in result.hits]
    assert len(chunk_ids) == len(set(chunk_ids))
    per_doc: dict[str, int] = {}
    for hit in result.hits:
        per_doc[hit.document_id] = per_doc.get(hit.document_id, 0) + 1
    assert max(per_doc.values()) <= 3


def test_missing_evidence_detection(retriever):
    snapshot_ids = {"banned-terms", "this-does-not-exist"}
    result = retriever.retrieve(
        "approval limit",
        filters=MetadataFilter(category="headsets", supplier="Monsoon Foods", negotiation_date=date(2026, 11, 10)),
        mandatory_document_ids=snapshot_ids,
    )
    assert "this-does-not-exist" in result.missing_evidence


def test_rrf_rank_fusion():
    fused = reciprocal_rank_fusion(
        [("a", 9.0), ("b", 8.0), ("c", 7.0)],
        [("b", 0.9), ("a", 0.8), ("d", 0.7)],
        k=60,
    )
    assert fused[0].chunk_id in {"a", "b"}
    assert {entry.chunk_id for entry in fused} == {"a", "b", "c", "d"}
    assert next(entry for entry in fused if entry.chunk_id == "b").rank_dense == 1


def test_retrieval_evaluator_macro_metrics(retriever):
    from src.retrieval.retrieval_evaluator import evaluate_retrieval

    report = evaluate_retrieval(retriever, "config/retrieval_queries_acme.yaml")
    assert report["query_count"] == 17
    assert report["macro_recall"] >= 0.9
    assert report["mrr"] >= 0.9
    assert report["hit_rate"] >= 0.9
