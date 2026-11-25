"""Ground-level validation of extraction output against the original message."""
from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from src.extraction.deterministic_extractor import (
    detect_billing_period,
    detect_gst,
    detect_unit_factor,
)
from src.extraction.extraction_schema import OfferExtraction, ValidationStatus
from src.knowledge.policy_snapshot import PolicySnapshot
from src.utils.text import latest_supplier_body


class ExtractionValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def validate_extraction_grounding(
    extraction: OfferExtraction, snapshot: PolicySnapshot
) -> ExtractionValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    latest = latest_supplier_body(extraction.source_message)
    if extraction.latest_message_text != latest:
        errors.append("latest_supplier_text_not_segmented_faithfully")
    if extraction.offer_text and extraction.offer_text not in extraction.source_message:
        errors.append("offer_text_not_verbatim_in_source")
    for candidate in extraction.price_candidates:
        if extraction.offer_text and candidate.raw_text not in extraction.offer_text:
            errors.append(f"price_candidate_not_grounded:{candidate.raw_text}")
    for candidate in extraction.previous_quote_candidates:
        if not extraction.previous_quote_text or candidate.raw_text not in extraction.previous_quote_text:
            errors.append(f"previous_quote_candidate_not_grounded:{candidate.raw_text}")
    if extraction.offer_text:
        if detect_gst(extraction.offer_text) != extraction.gst_treatment and extraction.gst_treatment != "excluded":
            errors.append("gst_treatment_not_reproducible")
        if detect_billing_period(extraction.offer_text) != extraction.billing_period:
            errors.append("billing_period_not_reproducible")
        factor, label, _ = detect_unit_factor(extraction.offer_text)
        if label and factor == 1:
            # Category-defined sub-units (ream, drum, pack) are checked by the
            # normalizer/verifier against the compiled category configuration.
            pass
        elif factor != extraction.quoted_unit_factor:
            errors.append("unit_factor_not_reproducible")
    if extraction.is_total and extraction.total_quantity is not None and extraction.total_quantity <= 0:
        errors.append("invalid_total_quantity")
    if extraction.validation_status == ValidationStatus.VALID and not extraction.price_candidates:
        if extraction.discount_base != "previous_quote":
            errors.append("valid_extraction_without_price")
    if extraction.discount_percent is not None and not (Decimal(0) <= extraction.discount_percent <= Decimal(100)):
        errors.append("discount_percent_out_of_range")
    if extraction.free_units is not None and extraction.paid_units is not None:
        if extraction.free_units < 0 or extraction.paid_units <= 0:
            errors.append("invalid_free_unit_ratio")
    return ExtractionValidationResult(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))
