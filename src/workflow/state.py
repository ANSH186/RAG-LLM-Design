"""Typed shared state for the LangGraph procurement workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from src.audit import AuditStore
from src.evidence.context_builder import ContextBuilder
from src.evidence.evidence_schema import EvidencePackage
from src.evidence.source_registry import SourceRegistry
from src.extraction.extraction_schema import NormalizedOffer, OfferExtraction
from src.extraction.extractor import ExtractionOrchestrator
from src.knowledge.policy_snapshot import PolicySnapshot
from src.knowledge.schema import CompiledKnowledge
from src.llm.provider import LLMConfig, LLMProvider
from src.observability.metrics import MetricsCollector
from src.observability.tracing import Tracer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.schemas import RetrievalResult
from src.verification.schemas import VerificationResult


class ProcurementState(TypedDict, total=False):
    run_id: str
    thread_row: dict[str, Any]
    thread_id: str
    category_slug: str
    supplier_name: str
    negotiation_date: str
    quantity: int
    round_number: int
    our_last_counter: float | None
    snapshot: PolicySnapshot
    pre_extraction: OfferExtraction
    pre_normalized: NormalizedOffer
    extraction: OfferExtraction
    normalized: NormalizedOffer
    extraction_outcome: dict[str, Any]
    security_injections: tuple
    security_banned: tuple
    early_rule: str | None
    retrieval_query: str
    retrieval: RetrievalResult
    evidence: EvidencePackage
    evidence_validation: dict[str, Any]
    context_hash: str
    decision: Any
    verification: VerificationResult
    verification_history: tuple[dict[str, Any], ...]
    attempts: int
    final_label: str
    safe_failure: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    timings_ms: dict[str, float]
    stage_trace: tuple[str, ...]
    audit_persisted: bool
    thread_started_perf: float


@dataclass
class PipelineServices:
    run_id: str
    knowledge: CompiledKnowledge
    registry: SourceRegistry
    retriever: HybridRetriever
    context_builder: ContextBuilder
    extraction_mode: str
    llm_provider: LLMProvider
    llm_config: LLMConfig
    metrics: MetricsCollector
    tracer: Tracer
    audit_store: AuditStore


def factor_resolver(services: PipelineServices, category_slug: str):
    category = services.knowledge.categories[category_slug]

    def resolve(label: str):
        return category.subunit_factors.get((label or "").lower())

    return resolve


def make_orchestrator(services: PipelineServices, category_slug: str) -> ExtractionOrchestrator:
    return ExtractionOrchestrator(
        category_factor_resolver=factor_resolver(services, category_slug),
        mode=services.extraction_mode,
        provider=services.llm_provider,
        llm_config=services.llm_config,
    )
