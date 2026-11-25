"""Source registry for official documents, chunks and content hashes."""
from __future__ import annotations

from src.errors import KnowledgeValidationError
from src.knowledge.schema import CompiledKnowledge, DocumentType
from src.retrieval.chunking import build_chunks
from src.retrieval.schemas import KnowledgeChunk


class SourceRegistry:
    def __init__(self, knowledge: CompiledKnowledge) -> None:
        self.knowledge = knowledge
        self.documents = {doc.document_id: doc for doc in knowledge.documents}
        self.chunks = build_chunks(knowledge)
        self.chunks_by_document: dict[str, list[KnowledgeChunk]] = {}
        for chunk in self.chunks:
            self.chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

    def get_document(self, document_id: str):
        try:
            return self.documents[document_id]
        except KeyError as exc:
            raise KnowledgeValidationError(f"unknown source document: {document_id}") from exc

    def get_chunks(self, document_id: str) -> list[KnowledgeChunk]:
        self.get_document(document_id)
        return self.chunks_by_document.get(document_id, [])

    def mandatory_policy_ids(self) -> list[str]:
        return [
            self.knowledge.policy_document_ids[key]
            for key in (
                "decision_procedure",
                "banned_terms",
                "security_policy",
                "units_and_currency",
                "approval_limits",
                "concession_schedule",
                "output_format",
            )
            if key in self.knowledge.policy_document_ids
        ]

    def supplier_document_id(self, supplier_id: str) -> str | None:
        supplier = self.knowledge.suppliers.get(supplier_id)
        return supplier.document_id if supplier else None

    def category_document_id(self, category_slug: str) -> str | None:
        category = self.knowledge.categories.get(category_slug)
        return category.document_id if category else None

    def amendment_document_ids(self, category_slug: str, negotiation_date) -> list[str]:
        return [
            amendment.document_id
            for amendment in self.knowledge.amendments
            if amendment.category_slug == category_slug
            and amendment.effective_from <= negotiation_date
        ]

    def validate_chunk_scope(
        self, chunk: KnowledgeChunk, *, category: str, supplier_id: str, negotiation_date
    ) -> None:
        if chunk.document_type == DocumentType.AMENDMENT.value:
            if chunk.category != category:
                raise KnowledgeValidationError(f"{chunk.document_id}: amendment category mismatch")
            if chunk.effective_from and chunk.effective_from > negotiation_date:
                raise KnowledgeValidationError(f"{chunk.document_id}: future amendment in evidence")
        if chunk.document_type == DocumentType.CATEGORY.value and chunk.category != category:
            raise KnowledgeValidationError(f"{chunk.document_id}: wrong category evidence")
        if chunk.document_type == DocumentType.SUPPLIER.value:
            supplier = self.knowledge.suppliers.get(supplier_id)
            if not supplier or supplier.document_id != chunk.document_id:
                raise KnowledgeValidationError(f"{chunk.document_id}: wrong supplier evidence")
