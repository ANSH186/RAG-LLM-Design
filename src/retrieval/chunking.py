"""Build retrieval chunks from compiled knowledge records."""
from __future__ import annotations

from src.knowledge.schema import CompiledKnowledge, DocumentType
from src.retrieval.schemas import KnowledgeChunk
from src.utils.text import normalise_slug


def build_chunks(knowledge: CompiledKnowledge) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for doc in knowledge.documents:
        for section in doc.sections:
            category = doc.category
            supplier = doc.supplier
            metadata = dict(doc.metadata or {})
            if doc.document_type == DocumentType.SUPPLIER:
                metadata["supplier_id"] = doc.document_id
            if doc.document_type == DocumentType.POLICY:
                metadata.setdefault("policy_kind", doc.document_id)
            if doc.document_type == DocumentType.AMENDMENT:
                metadata["amendment_id"] = doc.amendment_id or ""
                metadata["effective_from"] = doc.effective_from.isoformat() if doc.effective_from else ""
            if doc.document_type == DocumentType.PRECEDENT:
                metadata.setdefault("precedent_id", doc.document_id)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=section.section_id,
                    document_id=doc.document_id,
                    document_type=doc.document_type.value,
                    title=doc.title,
                    source_file=doc.source_file,
                    section_id=section.section_id,
                    heading=section.heading,
                    content=section.content,
                    content_hash=doc.content_hash,
                    category=category,
                    supplier=supplier,
                    effective_from=doc.effective_from,
                    effective_to=doc.effective_to,
                    amendment_id=doc.amendment_id,
                    metadata=metadata,
                )
            )
    return chunks


def applicable_chunks(
    chunks: list[KnowledgeChunk],
    *,
    category: str | None,
    supplier: str | None,
    negotiation_date,
    include_policies: bool = True,
    include_category: bool = True,
    include_supplier: bool = True,
    include_amendments: bool = True,
    include_precedents: bool = True,
    document_types: tuple[str, ...] = (),
    document_ids: tuple[str, ...] = (),
) -> list[KnowledgeChunk]:
    """Apply hard temporal and metadata filters before any scoring."""
    selected: list[KnowledgeChunk] = []
    allowed_types = set(document_types)
    allowed_ids = set(document_ids)
    for chunk in chunks:
        if allowed_types and chunk.document_type not in allowed_types:
            continue
        if allowed_ids and chunk.document_id not in allowed_ids:
            continue
        if chunk.document_type == "policy":
            if not include_policies:
                continue
        elif chunk.document_type == "category":
            if not include_category or chunk.category != category:
                continue
        elif chunk.document_type == "supplier":
            if not include_supplier or not supplier:
                continue
            if normalise_slug(chunk.supplier or "") != normalise_slug(supplier):
                continue
        elif chunk.document_type == "amendment":
            if not include_amendments or chunk.category != category:
                continue
            if chunk.effective_from and negotiation_date and chunk.effective_from > negotiation_date:
                continue
        elif chunk.document_type == "precedent":
            if not include_precedents:
                continue
            if chunk.category and chunk.category != category:
                continue
        else:
            continue
        if chunk.effective_to and negotiation_date and chunk.effective_to < negotiation_date:
            continue
        selected.append(chunk)
    return selected
