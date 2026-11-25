"""Retrieval quality evaluation against a hand-labelled query set."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import yaml

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.schemas import MetadataFilter


def evaluate_retrieval(
    retriever: HybridRetriever,
    query_set_path: str | Path,
    *,
    top_k: int | None = None,
    output_path: str | Path | None = None,
) -> dict:
    path = Path(query_set_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = data.get("queries", [])
    if not queries:
        raise ValueError(f"no queries in {path}")
    top_k = top_k or retriever.settings.retrieval.top_k
    results = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    hits_count = 0
    for query in queries:
        relevant = set(query.get("relevant_document_ids", []))
        filters = MetadataFilter(
            category=query.get("category"),
            supplier=query.get("supplier"),
            negotiation_date=date.fromisoformat(query["date"]) if query.get("date") else None,
        )
        result = retriever.retrieve(query["query"], filters=filters, top_k=top_k)
        retrieved_docs: list[str] = []
        for hit in result.hits:
            if hit.document_id not in retrieved_docs:
                retrieved_docs.append(hit.document_id)
        found = [doc for doc in retrieved_docs if doc in relevant]
        precision = len(found) / len(retrieved_docs) if retrieved_docs else 0.0
        recall = len(set(found)) / len(relevant) if relevant else 1.0
        first_rank = None
        for rank, hit in enumerate(result.hits, start=1):
            if hit.document_id in relevant:
                first_rank = rank
                break
        reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
        if found:
            hits_count += 1
        precision_values.append(precision)
        recall_values.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        results.append(
            {
                "query_id": query.get("id"),
                "query": query["query"],
                "relevant_document_ids": sorted(relevant),
                "retrieved_document_ids": retrieved_docs,
                "retrieved_chunk_ids": [hit.chunk_id for hit in result.hits],
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "reciprocal_rank": round(reciprocal_rank, 6),
                "hit": bool(found),
                "retrieval_confidence": result.confidence,
                "methods": list(result.methods),
                "missing_evidence": list(result.missing_evidence),
                "latency_ms": result.latency_ms,
            }
        )
    report = {
        "query_count": len(results),
        "top_k": top_k,
        "macro_precision": round(sum(precision_values) / len(precision_values), 6),
        "macro_recall": round(sum(recall_values) / len(recall_values), 6),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6),
        "hit_rate": round(hits_count / len(results), 6),
        "results": results,
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
