"""Independent evidence and source-integrity verification."""
from __future__ import annotations

from src.evidence.evidence_schema import EvidencePackage, TrustLevel
from src.evidence.provenance import build_provenance_map
from src.evidence.source_registry import SourceRegistry
from src.extraction.extraction_schema import OfferExtraction
from src.knowledge.schema import CompiledKnowledge
from src.utils.hashing import sha256_text
from src.utils.text import latest_supplier_body, normalise_text
from src.verification.schemas import VerificationCheck, VerificationIssue


def verify_evidence_package(
    *,
    package: EvidencePackage,
    extraction: OfferExtraction,
    knowledge: CompiledKnowledge,
    registry: SourceRegistry | None = None,
) -> tuple[list[VerificationIssue], list[VerificationCheck], bool]:
    registry = registry or SourceRegistry(knowledge)
    issues: list[VerificationIssue] = []
    snapshot = package.policy_snapshot

    if package.thread_id != extraction.thread_id:
        issues.append(
            VerificationIssue(
                code="evidence_thread_mismatch",
                detail="evidence package thread differs from extraction thread",
            )
        )
    if package.snapshot_id != snapshot.snapshot_id:
        issues.append(
            VerificationIssue(
                code="evidence_snapshot_mismatch",
                detail="evidence package snapshot id mismatch",
            )
        )
    if package.mandatory_missing:
        issues.append(
            VerificationIssue(
                code="mandatory_evidence_missing",
                detail="mandatory evidence documents are missing",
                actual=", ".join(package.mandatory_missing),
            )
        )

    all_items = list(package.official_evidence) + list(package.retrieved_evidence) + list(package.precedents)
    evidence_doc_ids = {item.document_id for item in all_items}
    for document_id in snapshot.source_documents:
        if document_id not in evidence_doc_ids:
            issues.append(
                VerificationIssue(
                    code="snapshot_source_not_in_evidence",
                    detail="a source used by the policy snapshot is absent from evidence",
                    actual=document_id,
                )
            )
    for key in (
        "decision_procedure",
        "banned_terms",
        "security_policy",
        "units_and_currency",
        "approval_limits",
        "concession_schedule",
        "output_format",
    ):
        document_id = knowledge.policy_document_ids.get(key)
        if document_id and document_id not in evidence_doc_ids:
            issues.append(
                VerificationIssue(
                    code="mandatory_policy_not_in_evidence",
                    detail="mandatory official policy is absent from evidence",
                    actual=document_id,
                )
            )

    registry_chunks = {chunk.chunk_id: chunk for chunk in registry.chunks}
    for item in all_items:
        document = registry.documents.get(item.document_id)
        if document is None:
            issues.append(
                VerificationIssue(
                    code="evidence_unknown_document",
                    detail="evidence references an unknown source document",
                    actual=item.document_id,
                )
            )
            continue
        if item.content_hash != document.content_hash:
            issues.append(
                VerificationIssue(
                    code="evidence_hash_mismatch",
                    detail="evidence content hash does not match the source document hash",
                    actual=item.document_id,
                )
            )
        if item.chunk_id:
            chunk = registry_chunks.get(item.chunk_id)
            if chunk is None:
                issues.append(
                    VerificationIssue(
                        code="evidence_unknown_chunk",
                        detail="evidence chunk is not in the source registry",
                        actual=item.chunk_id,
                    )
                )
            else:
                if chunk.content != item.content:
                    issues.append(
                        VerificationIssue(
                            code="evidence_content_mismatch",
                            detail="evidence content differs from the indexed source chunk",
                            actual=item.chunk_id,
                        )
                    )
                try:
                    registry.validate_chunk_scope(
                        chunk,
                        category=snapshot.category_slug,
                        supplier_id=snapshot.supplier_id,
                        negotiation_date=snapshot.negotiation_date,
                    )
                except Exception as exc:  # noqa: BLE001 - convert registry failure to audit issue
                    issues.append(
                        VerificationIssue(
                            code="evidence_scope_violation",
                            detail=str(exc),
                            actual=item.chunk_id,
                        )
                    )
        if item.effective_from and item.effective_from > snapshot.negotiation_date:
            issues.append(
                VerificationIssue(
                    code="future_amendment_in_evidence",
                    detail="evidence contains an amendment effective after the negotiation date",
                    actual=item.document_id,
                )
            )

    supplier = package.supplier_message
    if supplier.trust_level != TrustLevel.SUPPLIER_UNTRUSTED:
        issues.append(
            VerificationIssue(
                code="supplier_trust_level_invalid",
                detail="supplier message must be marked untrusted",
            )
        )
    if supplier.content_hash != sha256_text(supplier.raw_message):
        issues.append(
            VerificationIssue(
                code="supplier_message_hash_mismatch",
                detail="supplier message hash mismatch",
            )
        )
    if supplier.latest_supplier_text != latest_supplier_body(supplier.raw_message):
        issues.append(
            VerificationIssue(
                code="supplier_latest_message_mismatch",
                detail="latest supplier text was not segmented faithfully",
            )
        )
    normalized_thread = normalise_text(supplier.raw_message)
    normalized_latest = normalise_text(supplier.latest_supplier_text)
    for finding in supplier.injection_findings:
        if (
            finding.matched_text not in supplier.raw_message
            and finding.matched_text not in normalized_thread
        ):
            issues.append(
                VerificationIssue(
                    code="injection_finding_not_grounded",
                    detail="security finding text does not appear in the thread",
                    actual=finding.matched_text,
                )
            )
    for finding in supplier.banned_term_findings:
        if (
            finding.matched_text not in supplier.latest_supplier_text
            and finding.matched_text not in normalized_latest
        ):
            issues.append(
                VerificationIssue(
                    code="banned_finding_not_grounded",
                    detail="banned-term finding does not appear in the latest supplier message",
                    actual=finding.matched_text,
                )
            )

    if extraction.offer_text:
        if extraction.offer_text not in extraction.source_message:
            issues.append(
                VerificationIssue(
                    code="offer_text_not_in_source",
                    detail="extracted offer sentence is not present verbatim in the source message",
                )
            )
        if extraction.offer_span_start is not None:
            start = extraction.offer_span_start
            end = extraction.offer_span_end or (start + len(extraction.offer_text))
            if extraction.source_message[start:end] != extraction.offer_text:
                issues.append(
                    VerificationIssue(
                        code="offer_span_mismatch",
                        detail="offer source span does not reproduce the offer text",
                    )
                )
    for candidate in extraction.price_candidates:
        if candidate.raw_text not in extraction.offer_text and candidate.raw_text.strip() not in extraction.offer_text:
            issues.append(
                VerificationIssue(
                    code="price_candidate_not_grounded",
                    detail="extracted price token is not present in the offer sentence",
                    actual=candidate.raw_text,
                )
            )
    for candidate in extraction.previous_quote_candidates:
        if not extraction.previous_quote_text or candidate.raw_text not in extraction.previous_quote_text:
            issues.append(
                VerificationIssue(
                    code="previous_quote_candidate_not_grounded",
                    detail="previous-quote price token not found in previous supplier message",
                    actual=candidate.raw_text,
                )
            )

    provenance = build_provenance_map(snapshot, package)
    if not provenance.complete:
        issues.append(
            VerificationIssue(
                code="provenance_incomplete",
                detail="snapshot fields lack source evidence",
                actual=", ".join(provenance.missing_fields),
            )
        )
    if "UNTRUSTED SUPPLIER MESSAGE" not in package.context_text:
        issues.append(
            VerificationIssue(
                code="context_trust_boundary_missing",
                detail="context does not clearly mark supplier text as untrusted",
            )
        )
    if "TRUSTED OFFICIAL PROCUREMENT POLICY" not in package.context_text:
        issues.append(
            VerificationIssue(
                code="context_official_section_missing",
                detail="context does not include the trusted official-policy section",
            )
        )

    checks = [
        VerificationCheck(name="source_registry_integrity", passed=not any(i.code.startswith("evidence_") for i in issues)),
        VerificationCheck(name="temporal_scope", passed=not any(i.code in {"future_amendment_in_evidence", "evidence_scope_violation"} for i in issues)),
        VerificationCheck(name="supplier_text_untrusted", passed=not any(i.code.startswith("supplier_") or i.code == "context_trust_boundary_missing" for i in issues)),
        VerificationCheck(name="extraction_grounded", passed=not any(i.code.endswith("not_grounded") or i.code in {"offer_text_not_in_source", "offer_span_mismatch"} for i in issues)),
        VerificationCheck(name="provenance_complete", passed=provenance.complete),
    ]
    return issues, checks, provenance.complete
