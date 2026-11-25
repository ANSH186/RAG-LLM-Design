"""Hybrid retrieval orchestration: hard filters -> BM25 + dense -> RRF -> rerank."""
from __future__ import annotations

import time

from src.config import Settings, get_settings
from src.knowledge.schema import CompiledKnowledge
from src.retrieval.bm25 import BM25Index
from src.retrieval.chunking import applicable_chunks, build_chunks
from src.retrieval.rank_fusion import reciprocal_rank_fusion
from src.retrieval.reranker import LexicalFeatureReranker
from src.retrieval.schemas import (
    KnowledgeChunk,
    MetadataFilter,
    RetrievalHit,
    RetrievalResult,
)
from src.retrieval.vector_store import DenseVectorStore, create_embedder
from src.utils.text import tokenize

# Mandatory policy documents for the general decision procedure. These are always
# eligible regardless of category and are checked for evidence coverage.
MANDATORY_POLICY_KEYS = (
    "decision_procedure",
    "banned_terms",
    "security_policy",
    "units_and_currency",
    "approval_limits",
    "concession_schedule",
    "output_format",
)


class HybridRetriever:
    def __init__(
        self,
        knowledge: CompiledKnowledge,
        *,
        settings: Settings | None = None,
        chunks: list[KnowledgeChunk] | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.settings = settings or get_settings()
        self.chunks = chunks or build_chunks(knowledge)
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.bm25 = BM25Index(self.chunks)
        self.embedder = create_embedder(
            self.settings.retrieval.embedding_backend,
            dimension=self.settings.retrieval.embedding_dim,
        )
        self.vector_store = DenseVectorStore.build(self.chunks, self.embedder)
        self.reranker = (
            LexicalFeatureReranker() if self.settings.retrieval.use_reranker else None
        )
        self.policy_doc_ids = {
            key: knowledge.policy_document_ids[key]
            for key in MANDATORY_POLICY_KEYS
            if key in knowledge.policy_document_ids
        }

    def mandatory_document_ids(self, snapshot) -> set[str]:
        """Documents that must be represented as thread-specific evidence.

        Global policy documents are loaded from the source registry by the evidence
        layer rather than forced into the top-k. This avoids an impossible requirement
        of fitting every global policy into a short retrieval window.
        """
        ids = set(snapshot.source_documents)
        category = self.knowledge.categories.get(snapshot.category_slug)
        if category:
            ids.add(category.document_id)
        supplier = self.knowledge.suppliers.get(snapshot.supplier_id)
        if supplier:
            ids.add(supplier.document_id)
        return ids

    def retrieve(
        self,
        query: str,
        *,
        filters: MetadataFilter,
        top_k: int | None = None,
        mandatory_document_ids: set[str] | None = None,
        candidate_limit: int = 60,
    ) -> RetrievalResult:
        started = time.perf_counter()
        top_k = top_k or self.settings.retrieval.top_k
        allowed = applicable_chunks(
            self.chunks,
            category=filters.category,
            supplier=filters.supplier,
            negotiation_date=filters.negotiation_date,
            include_policies=filters.include_policies,
            include_category=filters.include_category,
            include_supplier=filters.include_supplier,
            include_amendments=filters.include_amendments,
            include_precedents=filters.include_precedents,
            document_types=filters.document_types,
            document_ids=filters.document_ids,
        )
        allowed_ids = {chunk.chunk_id for chunk in allowed}
        candidate_limit = max(candidate_limit, top_k)
        bm25_hits = self.bm25.search(query, allowed_ids=allowed_ids, top_k=candidate_limit)
        query_vec = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search(
            query_vec, allowed_ids=allowed_ids, top_k=candidate_limit
        )
        fused = reciprocal_rank_fusion(
            [(hit.chunk_id, hit.score) for hit in bm25_hits],
            dense_hits,
            k=self.settings.retrieval.rrf_k,
            bm25_weight=self.settings.retrieval.bm25_weight,
            dense_weight=self.settings.retrieval.dense_weight,
        )
        candidates: list[tuple[KnowledgeChunk, float]] = [
            (self.by_id[entry.chunk_id], entry.score) for entry in fused
        ]
        used_methods: set[str] = {"bm25", "dense"} if dense_hits else {"bm25"}
        rerank_scores: dict[str, float] = {}
        if self.reranker and candidates:
            reranked = self.reranker.rerank(
                query, candidates, top_n=min(self.settings.retrieval.rerank_top_n, len(candidates))
            )
            candidates = [(chunk, fused_score) for chunk, fused_score, _ in reranked]
            rerank_scores = {chunk.chunk_id: score for chunk, _, score in reranked}
            used_methods.add("reranker")

        # Duplicate removal and source diversity: at most three sections per document.
        hits: list[RetrievalHit] = []
        seen_content: set[str] = set()
        per_document: dict[str, int] = {}
        entry_by_id = {entry.chunk_id: entry for entry in fused}
        for chunk, fused_score in candidates:
            if chunk.content_hash in seen_content:
                continue
            if per_document.get(chunk.document_id, 0) >= 3:
                continue
            seen_content.add(chunk.content_hash)
            per_document[chunk.document_id] = per_document.get(chunk.document_id, 0) + 1
            entry = entry_by_id.get(chunk.chunk_id)
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    section_id=chunk.section_id,
                    source_file=chunk.source_file,
                    source_section=chunk.heading,
                    title=chunk.title,
                    content=chunk.content,
                    document_type=chunk.document_type,
                    category=chunk.category,
                    supplier=chunk.supplier,
                    effective_from=chunk.effective_from,
                    bm25_score=entry.bm25_score if entry else None,
                    dense_score=entry.dense_score if entry else None,
                    fused_score=fused_score,
                    rerank_score=rerank_scores.get(chunk.chunk_id),
                    relevance_score=rerank_scores.get(chunk.chunk_id, fused_score),
                    retrieval_methods=tuple(sorted(entry.methods)) if entry else tuple(sorted(used_methods)),
                    rank_bm25=entry.rank_bm25 if entry else None,
                    rank_dense=entry.rank_dense if entry else None,
                    content_hash=chunk.content_hash,
                )
            )
            if len(hits) >= top_k:
                break

        retrieved_doc_ids = {hit.document_id for hit in hits}
        mandatory = mandatory_document_ids or set()
        missing = tuple(sorted(mandatory - retrieved_doc_ids))
        confidence = self._confidence(query, hits, mandatory, missing)
        elapsed = (time.perf_counter() - started) * 1000.0
        return RetrievalResult(
            query=query,
            hits=tuple(hits),
            candidate_count=len(self.chunks),
            filtered_count=len(allowed),
            methods=tuple(sorted(used_methods)),
            confidence=confidence,
            missing_evidence=missing,
            latency_ms=round(elapsed, 3),
            filters=filters,
        )

    def _confidence(
        self,
        query: str,
        hits: list[RetrievalHit],
        mandatory: set[str],
        missing: tuple[str, ...],
    ) -> float:
        if not hits:
            return 0.0
        query_tokens = set(tokenize(query))
        top_text = " ".join(hit.content for hit in hits[:3]).lower()
        coverage = (
            len(query_tokens & set(tokenize(top_text))) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        max_possible_rrf = 2.0 / (self.settings.retrieval.rrf_k + 1.0)
        top_rrf = hits[0].fused_score / max_possible_rrf if max_possible_rrf else 0.0
        top_rrf = max(0.0, min(1.0, top_rrf))
        mandatory_score = 1.0 if not mandatory else (len(mandatory) - len(missing)) / len(mandatory)
        confidence = 0.45 * coverage + 0.35 * top_rrf + 0.20 * mandatory_score
        return round(max(0.0, min(1.0, confidence)), 4)
