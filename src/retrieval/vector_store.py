"""Dense embedding backends.

The default backend is a deterministic hashed n-gram embedding. It runs offline,
needs no GPU, and captures lexical/semantic neighbourhoods without a heavy model.
An optional SentenceTransformers backend can be enabled explicitly when the
runtime has a working PyTorch installation; it is never silently required.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

from src.utils.text import normalise_text, tokenize


class Embedder(Protocol):
    name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _stable_hash(feature: str) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    index = value % (2**32)
    sign = 1.0 if (value >> 33) & 1 else -1.0
    return index, sign


class HashNgramEmbedder:
    """Hashed word/character n-gram embedding with L2 normalisation.

    This is a real dense vector representation (fixed-width, non-sparse), but it is
    not a pretrained semantic encoder. It is intentionally described as such in the
    documentation: BM25 carries exact policy terminology; this backend adds robust
    fuzzy/word-order similarity when a neural encoder is unavailable.
    """

    name = "hash-ngram"

    def __init__(self, dimension: int = 512) -> None:
        if dimension < 64:
            raise ValueError("embedding dimension must be at least 64")
        self.dimension = dimension

    def _features(self, text: str) -> list[tuple[str, float]]:
        normalized = normalise_text(text).lower()
        words = tokenize(normalized)
        features: list[tuple[str, float]] = []
        for word in words:
            features.append((f"w:{word}", 1.0))
            padded = f"<{word}>"
            for n in (3, 4):
                for i in range(max(0, len(padded) - n + 1)):
                    features.append((f"c:{padded[i:i+n]}", 0.35))
        for left, right in zip(words, words[1:]):
            features.append((f"b:{left}_{right}", 0.8))
        # Whole-phrase character shingles help with amounts and contract wording.
        compact = re.sub(r"\s+", " ", normalized)
        for n in (8, 12):
            for i in range(0, max(0, len(compact) - n + 1), max(1, n // 2)):
                features.append((f"p:{compact[i:i+n]}", 0.15))
        return features

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        counts: dict[str, int] = {}
        for feature, weight in self._features(text):
            counts[feature] = counts.get(feature, 0) + 1
            weight_total = weight
            if counts[feature] > 1:
                weight_total *= 1.0 + math.log(counts[feature])
            index, sign = _stable_hash(feature)
            vector[index % self.dimension] += sign * weight_total
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class SentenceTransformerEmbedder:
    """Optional neural encoder; loaded lazily and only when explicitly configured."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on host runtime
            raise RuntimeError(
                "sentence-transformers is not usable in this environment; "
                "use RETRIEVAL_EMBEDDING_BACKEND=hash or install a working PyTorch runtime"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(
            self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )


class DenseVectorStore:
    """Small in-process vector index with mandatory metadata ids.

    A FAISS/Chroma dependency is not justified for ~100 playbook chunks. The store
    persists as a NumPy matrix plus a JSON id map, which is reproducible and portable.
    """

    def __init__(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError("chunk id count does not match vector count")
        self.chunk_ids = list(chunk_ids)
        self.id_to_pos = {chunk_id: i for i, chunk_id in enumerate(self.chunk_ids)}
        self.vectors = vectors.astype(np.float32, copy=False)

    @classmethod
    def build(cls, chunks, embedder: Embedder) -> DenseVectorStore:
        texts = [
            " | ".join(
                [
                    chunk.title,
                    chunk.heading,
                    chunk.category or "",
                    chunk.supplier or "",
                    chunk.content,
                ]
            )
            for chunk in chunks
        ]
        vectors = embedder.embed_documents(texts)
        return cls([chunk.chunk_id for chunk in chunks], vectors)

    def search(
        self, query_vector: np.ndarray, *, allowed_ids: set[str] | None = None, top_k: int = 50
    ) -> list[tuple[str, float]]:
        if self.vectors.shape[0] == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError("query embedding dimension mismatch")
        scores = self.vectors @ query
        candidates = range(len(self.chunk_ids))
        if allowed_ids is not None:
            candidates = [i for i in candidates if self.chunk_ids[i] in allowed_ids]
        ranked = sorted(
            ((self.chunk_ids[i], float(scores[i])) for i in candidates),
            key=lambda item: (-item[1], item[0]),
        )
        return [item for item in ranked[:top_k] if item[1] > 0]

    def save(self, matrix_path, id_map_path) -> None:  # pragma: no cover - exercised by scripts
        import json

        np.save(matrix_path, self.vectors)
        id_map_path.write_text(json.dumps(self.chunk_ids, indent=2), encoding="utf-8")


def create_embedder(backend: str, dimension: int = 512, model_name: str | None = None) -> Embedder:
    key = backend.strip().lower()
    if key in {"hash", "hashed", "hash-ngram"}:
        return HashNgramEmbedder(dimension=dimension)
    if key in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"unknown embedding backend: {backend!r}")
