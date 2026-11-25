"""Retrieval contracts shared by lexical, dense and hybrid search."""
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DocumentTypeFilter(str, Enum):
    POLICY = "policy"
    CATEGORY = "category"
    SUPPLIER = "supplier"
    AMENDMENT = "amendment"
    PRECEDENT = "precedent"


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    document_type: str
    title: str
    source_file: str
    section_id: str
    heading: str
    content: str
    content_hash: str
    category: str | None = None
    supplier: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    amendment_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.document_id}#{self.section_id.split('#')[-1]}"


class MetadataFilter(BaseModel):
    """Hard filters. Semantic similarity can never relax these."""

    model_config = ConfigDict(frozen=True)

    category: str | None = None
    supplier: str | None = None
    negotiation_date: date | None = None
    include_policies: bool = True
    include_category: bool = True
    include_supplier: bool = True
    include_amendments: bool = True
    include_precedents: bool = True
    document_types: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()


class RetrievalHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    section_id: str
    source_file: str
    source_section: str
    title: str
    content: str
    document_type: str
    category: str | None = None
    supplier: str | None = None
    effective_from: date | None = None
    bm25_score: float | None = None
    dense_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    relevance_score: float = 0.0
    retrieval_methods: tuple[str, ...] = ()
    rank_bm25: int | None = None
    rank_dense: int | None = None
    content_hash: str


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    hits: tuple[RetrievalHit, ...]
    candidate_count: int
    filtered_count: int
    methods: tuple[str, ...]
    confidence: float = Field(ge=0, le=1)
    missing_evidence: tuple[str, ...] = ()
    latency_ms: float
    filters: MetadataFilter = Field(default_factory=MetadataFilter)
