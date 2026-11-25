"""Validation helpers for compiled knowledge and source fidelity."""
from __future__ import annotations

from pathlib import Path

from src.errors import KnowledgeValidationError
from src.knowledge.ingestion import verify_source_hashes
from src.knowledge.policy_compiler import validate_compiled_knowledge
from src.knowledge.schema import CompiledKnowledge


def validate_knowledge_against_sources(
    knowledge: CompiledKnowledge, project_root: Path
) -> None:
    validate_compiled_knowledge(knowledge)
    problems = verify_source_hashes(knowledge, project_root)
    if problems:
        raise KnowledgeValidationError("; ".join(problems))


def assert_no_duplicate_source_text(knowledge: CompiledKnowledge) -> None:
    hashes: set[str] = set()
    for doc in knowledge.documents:
        if doc.content_hash in hashes:
            raise KnowledgeValidationError(f"{doc.document_id}: duplicate document content")
        hashes.add(doc.content_hash)


def assert_mandatory_rule_fragments(knowledge: CompiledKnowledge) -> None:
    """Check that every decision-procedure rule is present in the source text.

    This guards against silently missing a policy document while keeping the
    deterministic implementation independent of an LLM's summary.
    """
    doc_id = knowledge.policy_document_ids["decision_procedure"]
    doc = next(d for d in knowledge.documents if d.document_id == doc_id)
    content = doc.content.lower()
    fragments = [
        "missing information",
        "blacklisted supplier",
        "prompt injection",
        "banned contract terms",
        "price decision",
        "approval limit",
        "probation",
        "later amendments replace earlier ones",
        "supplier contract",
    ]
    missing = [fragment for fragment in fragments if fragment not in content]
    if missing:
        raise KnowledgeValidationError(f"decision procedure missing fragments: {missing}")
    banned_id = knowledge.policy_document_ids["banned_terms"]
    banned = next(d for d in knowledge.documents if d.document_id == banned_id).content.lower()
    for fragment in ("automatic renewal", "advance", "unlimited liability", "exclusivity"):
        if fragment not in banned:
            raise KnowledgeValidationError(f"banned-terms document missing {fragment!r}")
