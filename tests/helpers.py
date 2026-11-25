"""Shared test helpers for extraction, decision and verification."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

from src.decision.decision_engine import decide
from src.decision.security_rules import scan_security
from src.evidence.context_builder import ContextBuilder
from src.evidence.source_registry import SourceRegistry
from src.extraction.deterministic_extractor import DeterministicExtractor
from src.extraction.extraction_schema import NormalizedOffer, OfferExtraction
from src.extraction.normalizer import OfferNormalizer
from src.knowledge.policy_snapshot import PolicySnapshot
from src.knowledge.schema import CompiledKnowledge
from src.knowledge.temporal_resolver import resolve_policy_snapshot
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.schemas import MetadataFilter
from src.verification.decision_verifier import verify_decision
from src.verification.schemas import VerificationResult


@dataclass
class Evaluated:
    snapshot: PolicySnapshot
    extraction: OfferExtraction
    normalized: NormalizedOffer
    decision: object


def snapshot_for(
    knowledge: CompiledKnowledge,
    *,
    category: str,
    supplier: str,
    date: str,
    thread_id: str = "TEST",
) -> PolicySnapshot:
    return resolve_policy_snapshot(
        knowledge,
        thread_id=thread_id,
        category_slug=category,
        supplier_name=supplier,
        negotiation_date=date,
    )


def evaluate_message(
    knowledge: CompiledKnowledge,
    *,
    category: str,
    supplier: str,
    date: str,
    message: str,
    quantity: int,
    round_number: int,
    thread_id: str = "TEST",
) -> Evaluated:
    snapshot = snapshot_for(
        knowledge, category=category, supplier=supplier, date=date, thread_id=thread_id
    )
    category_obj = knowledge.categories[category]
    resolver = lambda label, c=category_obj: c.subunit_factors.get((label or "").lower())
    extraction = DeterministicExtractor(category_factor_resolver=resolver).extract(
        thread_id=thread_id, source_message=message, snapshot=snapshot
    )
    normalized = OfferNormalizer(snapshot, factor_resolver=resolver).normalize(
        extraction, quantity=quantity
    )
    injections, banned = scan_security(message)
    decision = decide(
        thread_id=thread_id,
        category=category,
        supplier=supplier,
        round_number=round_number,
        quantity=quantity,
        extraction=extraction,
        normalized=normalized,
        snapshot=snapshot,
        injection_findings=injections,
        banned_term_findings=banned,
    )
    return Evaluated(snapshot=snapshot, extraction=extraction, normalized=normalized, decision=decision)


def normalized_price(
    knowledge: CompiledKnowledge,
    *,
    category: str,
    supplier: str,
    date: str,
    message: str,
    quantity: int,
) -> object:
    evaluated = evaluate_message(
        knowledge,
        category=category,
        supplier=supplier,
        date=date,
        message=message,
        quantity=quantity,
        round_number=1,
    )
    return evaluated.normalized


def evaluate_with_evidence(
    knowledge: CompiledKnowledge,
    retriever: HybridRetriever,
    registry: SourceRegistry,
    context_builder: ContextBuilder,
    *,
    category: str,
    supplier: str,
    date: str,
    message: str,
    quantity: int,
    round_number: int,
    thread_id: str = "TEST",
) -> tuple[Evaluated, VerificationResult]:
    evaluated = evaluate_message(
        knowledge,
        category=category,
        supplier=supplier,
        date=date,
        message=message,
        quantity=quantity,
        round_number=round_number,
        thread_id=thread_id,
    )
    filter_value = MetadataFilter(
        category=category, supplier=supplier, negotiation_date=date_type.fromisoformat(date)
    )
    query = f"{category} {supplier} {message[:300]}"
    retrieval = retriever.retrieve(
        query,
        filters=filter_value,
        mandatory_document_ids=retriever.mandatory_document_ids(evaluated.snapshot),
    )
    package = context_builder.build(
        thread_id=thread_id,
        source_message=message,
        snapshot=evaluated.snapshot,
        retrieval=retrieval,
        injection_findings=evaluated.decision.injection_findings,
        banned_term_findings=evaluated.decision.banned_term_findings,
    )
    verification = verify_decision(
        decision=evaluated.decision,
        extraction=evaluated.extraction,
        normalized=evaluated.normalized,
        snapshot=evaluated.snapshot,
        knowledge=knowledge,
        evidence_package=package,
        registry=registry,
    )
    return evaluated, verification
