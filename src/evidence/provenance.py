"""Provenance maps connecting policy values and decisions to source documents."""
from __future__ import annotations

from src.evidence.evidence_schema import EvidencePackage, ProvenanceMap
from src.knowledge.policy_snapshot import PolicySnapshot
from src.knowledge.schema import CompiledKnowledge

SNAPSHOT_FIELDS = (
    "target_price",
    "walkaway_price",
    "approval_limit",
    "gst_rate",
    "negotiation_rounds",
    "concession_beta",
    "currency_rates",
    "supplier_status",
)
RULE_SOURCE_KEYS = {
    "MISSING_INFORMATION": ("units_and_currency",),
    "BLACKLISTED_SUPPLIER": ("units_and_currency",),
    "PROMPT_INJECTION": ("security_policy",),
    "BANNED_TERM": ("banned_terms",),
    "PRICE_DECISION": ("concession_schedule", "units_and_currency", "decision_procedure"),
    "APPROVAL_LIMIT": ("approval_limits", "decision_procedure"),
}


def build_provenance_map(snapshot: PolicySnapshot, package: EvidencePackage) -> ProvenanceMap:
    available_docs = {item.document_id for item in package.official_evidence} | {
        item.document_id for item in package.retrieved_evidence
    }
    available_docs.update(item.document_id for item in package.precedents)
    field_sources: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    for field in SNAPSHOT_FIELDS:
        provenance = snapshot.field_provenance.get(field, ())
        doc_ids = tuple(dict.fromkeys(source.document_id for source in provenance))
        if field == "supplier_status" and not doc_ids:
            doc_ids = (snapshot.supplier_id,)
        if field == "currency_rates" and not doc_ids:
            doc_ids = ("units-and-currency",)
        field_sources[field] = doc_ids
        if not doc_ids or any(doc_id not in available_docs for doc_id in doc_ids):
            missing.append(field)
    return ProvenanceMap(
        thread_id=snapshot.thread_id,
        snapshot_id=snapshot.snapshot_id,
        field_sources=field_sources,
        missing_fields=tuple(missing),
        complete=not missing,
    )


def source_documents_for_rule(
    rule_name: str, snapshot: PolicySnapshot, knowledge: CompiledKnowledge
) -> set[str]:
    docs: set[str] = set(snapshot.source_documents)
    category = knowledge.categories.get(snapshot.category_slug)
    if category:
        docs.add(category.document_id)
    supplier = knowledge.suppliers.get(snapshot.supplier_id)
    if supplier:
        docs.add(supplier.document_id)
    for key in RULE_SOURCE_KEYS.get(rule_name, ()):
        document_id = knowledge.policy_document_ids.get(key)
        if document_id:
            docs.add(document_id)
    return {doc for doc in docs if doc}
