"""Build compiled knowledge and retrieval indexes for an environment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_environment
from src.knowledge.ingestion import ingest_knowledge
from src.knowledge.validation import (
    assert_mandatory_rule_fragments,
    validate_knowledge_against_sources,
)
from src.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default=None)
    parser.add_argument("--write-index-manifest", action="store_true")
    args = parser.parse_args()
    env = get_environment(args.environment)
    report = ingest_knowledge(env)
    knowledge = report.knowledge
    validate_knowledge_against_sources(knowledge, ROOT)
    assert_mandatory_rule_fragments(knowledge)
    retriever = HybridRetriever(knowledge)
    manifest = {
        "environment": env.code,
        "compiled": report.summary(),
        "chunks": len(retriever.chunks),
        "embedding_backend": retriever.embedder.name,
        "embedding_dimension": retriever.embedder.dimension,
        "retrieval": {
            "top_k": retriever.settings.retrieval.top_k,
            "rrf_k": retriever.settings.retrieval.rrf_k,
            "bm25_weight": retriever.settings.retrieval.bm25_weight,
            "dense_weight": retriever.settings.retrieval.dense_weight,
            "reranker": retriever.settings.retrieval.use_reranker,
        },
    }
    index_dir = env.resolved(env.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    retriever.vector_store.save(
        index_dir / "dense_vectors.npy", index_dir / "chunk_ids.json"
    )
    (index_dir / "chunks.json").write_text(
        json.dumps(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "category": chunk.category,
                    "supplier": chunk.supplier,
                    "source_file": chunk.source_file,
                    "section": chunk.heading,
                    "content_hash": chunk.content_hash,
                }
                for chunk in retriever.chunks
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (index_dir / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
