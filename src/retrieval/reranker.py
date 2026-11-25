"""Optional reranking.

The default reranker is a lightweight feature reranker, not a transformer. It uses
query-term coverage, exact phrase hits and structural boosts. It only reorders the
already metadata-filtered candidate set; it cannot reintroduce a temporally invalid
document. A neural cross-encoder can be enabled through the optional UI/dependency
extras if a working PyTorch runtime is available.
"""
from __future__ import annotations

from src.retrieval.schemas import KnowledgeChunk
from src.utils.text import tokenize


class LexicalFeatureReranker:
    name = "lexical-feature-reranker"

    def score(self, query: str, chunk: KnowledgeChunk) -> float:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return 0.0
        doc_tokens = set(tokenize(f"{chunk.title} {chunk.heading} {chunk.content}"))
        coverage = len(query_tokens & doc_tokens) / len(query_tokens)
        phrase = 0.0
        lowered_query = query.lower()
        lowered_doc = chunk.content.lower()
        for phrase_length in range(2, 5):
            words = lowered_query.split()
            for i in range(max(0, len(words) - phrase_length + 1)):
                candidate = " ".join(words[i : i + phrase_length])
                if len(candidate) > 8 and candidate in lowered_doc:
                    phrase += 0.05
        boost = 0.0
        if chunk.document_type in {"policy", "category"}:
            boost += 0.05
        if chunk.category and chunk.category in lowered_query:
            boost += 0.08
        if chunk.supplier and chunk.supplier.lower() in lowered_query:
            boost += 0.08
        return coverage + min(phrase, 0.25) + boost

    def rerank(
        self, query: str, candidates: list[tuple[KnowledgeChunk, float]], top_n: int
    ) -> list[tuple[KnowledgeChunk, float, float]]:
        scored: list[tuple[KnowledgeChunk, float, float]] = []
        for chunk, fused_score in candidates:
            rerank_score = self.score(query, chunk)
            scored.append((chunk, fused_score, rerank_score))
        scored.sort(
            key=lambda item: (-(0.65 * item[2] + 0.35 * item[1]), item[0].chunk_id)
        )
        return scored[:top_n]


class CrossEncoderReranker:  # pragma: no cover - optional dependency
    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # type: ignore

        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[tuple[KnowledgeChunk, float]], top_n: int
    ) -> list[tuple[KnowledgeChunk, float, float]]:
        pairs = [(query, chunk.content) for chunk, _ in candidates]
        scores = self._model.predict(pairs)
        rescored = [
            (chunk, fused, float(score))
            for (chunk, fused), score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda item: (-item[2], item[0].chunk_id))
        return rescored[:top_n]


def create_reranker(kind: str = "lexical"):
    if kind in {"lexical", "feature", "none"}:
        return LexicalFeatureReranker() if kind != "none" else None
    if kind in {"cross-encoder", "cross_encoder"}:
        return CrossEncoderReranker()
    raise ValueError(f"unknown reranker: {kind}")
