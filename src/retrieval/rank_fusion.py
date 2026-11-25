"""Reciprocal Rank Fusion and rank utilities.

RRF is chosen because BM25 and the hashed dense encoder produce scores on different
scales. Rank-based fusion needs no score calibration, is deterministic, and is robust
when one retriever fails on a rare query.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusionEntry:
    chunk_id: str
    score: float = 0.0
    bm25_score: float | None = None
    dense_score: float | None = None
    rank_bm25: int | None = None
    rank_dense: int | None = None
    methods: set[str] = field(default_factory=set)


def reciprocal_rank_fusion(
    bm25_ranking: list[tuple[str, float]],
    dense_ranking: list[tuple[str, float]],
    *,
    k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[FusionEntry]:
    if k < 1:
        raise ValueError("RRF k must be positive")
    entries: dict[str, FusionEntry] = {}
    for rank, (chunk_id, score) in enumerate(bm25_ranking, start=1):
        entry = entries.setdefault(chunk_id, FusionEntry(chunk_id=chunk_id))
        entry.score += bm25_weight / (k + rank)
        entry.bm25_score = score
        entry.rank_bm25 = rank
        entry.methods.add("bm25")
    for rank, (chunk_id, score) in enumerate(dense_ranking, start=1):
        entry = entries.setdefault(chunk_id, FusionEntry(chunk_id=chunk_id))
        entry.score += dense_weight / (k + rank)
        entry.dense_score = score
        entry.rank_dense = rank
        entry.methods.add("dense")
    return sorted(entries.values(), key=lambda e: (-e.score, e.chunk_id))
