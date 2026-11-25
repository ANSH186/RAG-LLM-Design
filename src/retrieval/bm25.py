"""Dependency-free Okapi BM25 lexical retrieval."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from src.retrieval.schemas import KnowledgeChunk
from src.utils.text import tokenize


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0 or not (0 <= b <= 1):
            raise ValueError("invalid BM25 parameters")
        self.k1 = k1
        self.b = b
        self.chunks = list(chunks)
        self.chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        self.id_to_pos = {chunk_id: i for i, chunk_id in enumerate(self.chunk_ids)}
        self.token_counts: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        for chunk in self.chunks:
            text = " ".join(
                [
                    chunk.title,
                    chunk.heading,
                    chunk.category or "",
                    chunk.supplier or "",
                    chunk.content,
                ]
            )
            tokens = tokenize(text)
            self.token_counts.append(Counter(tokens))
            self.doc_lengths.append(len(tokens))
        self.n_docs = len(self.chunks)
        self.avgdl = (sum(self.doc_lengths) / self.n_docs) if self.n_docs else 0.0
        df: Counter[str] = Counter()
        for counts in self.token_counts:
            for token in counts:
                df[token] += 1
        self.idf = {
            token: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def search(self, query: str, *, allowed_ids: set[str] | None = None, top_k: int = 50) -> list[BM25Hit]:
        query_tokens = tokenize(query)
        if not query_tokens or self.n_docs == 0:
            return []
        query_counts = Counter(query_tokens)
        scores: list[tuple[str, float]] = []
        for index, counts in enumerate(self.token_counts):
            chunk_id = self.chunk_ids[index]
            if allowed_ids is not None and chunk_id not in allowed_ids:
                continue
            score = 0.0
            doc_len = self.doc_lengths[index]
            norm = self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl else 0.0))
            for token, qtf in query_counts.items():
                tf = counts.get(token, 0)
                if not tf:
                    continue
                idf = self.idf.get(token, 0.0)
                score += idf * ((tf * (self.k1 + 1.0)) / (tf + norm)) * qtf
            if score > 0:
                scores.append((chunk_id, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return [BM25Hit(chunk_id=cid, score=score) for cid, score in scores[:top_k]]
