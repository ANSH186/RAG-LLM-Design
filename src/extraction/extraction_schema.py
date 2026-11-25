"""Schemas for supplier-offer extraction and independent verification."""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExtractionMethod(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    HYBRID = "hybrid"
    NONE = "none"


class ValidationStatus(str, Enum):
    VALID = "VALID"
    INCOMPLETE = "INCOMPLETE"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


class PriceCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    amount: Decimal
    currency: str | None
    raw_text: str
    start: int
    end: int
    numeric_form: Literal["digits", "words"] = "digits"


class OfferExtraction(BaseModel):
    """What the supplier's latest message offers, before policy normalisation."""

    model_config = ConfigDict(frozen=True)

    thread_id: str
    extraction_method: ExtractionMethod
    extraction_confidence: float = Field(ge=0, le=1)
    validation_status: ValidationStatus
    source_message: str
    latest_message_text: str
    offer_text: str | None = None
    offer_span_start: int | None = None
    offer_span_end: int | None = None
    price_candidates: tuple[PriceCandidate, ...] = ()
    selection_rule: Literal["single", "range_high", "tier_by_quantity", "correction", "none"] = "none"
    selected_candidate_index: int | None = None
    tier_threshold: int | None = None
    is_correction: bool = False
    superseded_amount: Decimal | None = None
    superseded_currency: str | None = None
    is_total: bool = False
    total_quantity: int | None = None
    quoted_unit_factor: Decimal = Decimal(1)
    quoted_unit_label: str | None = None
    billing_period: Literal["month", "year"] | None = None
    gst_treatment: Literal["included", "excluded", "unspecified"] = "unspecified"
    discount_percent: Decimal | None = None
    discount_base: Literal["list_price", "previous_quote"] | None = None
    volume_discount_percent: Decimal | None = None
    volume_threshold: int | None = None
    volume_applies: bool | None = None
    free_units: int | None = None
    paid_units: int | None = None
    previous_quote_text: str | None = None
    previous_quote_candidates: tuple[PriceCandidate, ...] = ()
    previous_quote_unit_factor: Decimal = Decimal(1)
    previous_quote_billing_period: Literal["month", "year"] | None = None
    previous_quote_gst_treatment: Literal["included", "excluded", "unspecified"] = "unspecified"
    payment_terms: str | None = None
    delivery_terms: str | None = None
    warranty_terms: str | None = None
    ambiguities: tuple[str, ...] = ()
    preliminary_missing_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    source_locations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> OfferExtraction:
        if self.validation_status == ValidationStatus.VALID:
            if not self.price_candidates and self.discount_base != "previous_quote":
                raise ValueError("VALID extraction must include a price or a previous-quote reference")
        return self

    @property
    def selected_candidate(self) -> PriceCandidate | None:
        if self.selected_candidate_index is None:
            if len(self.price_candidates) == 1:
                return self.price_candidates[0]
            return None
        return self.price_candidates[self.selected_candidate_index]


class NormalizationStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: str
    detail: str
    input_value: str | None = None
    output_value: str | None = None
    source_document_id: str | None = None


class NormalizedOffer(BaseModel):
    model_config = ConfigDict(frozen=True)

    is_usable: bool
    normalized_price_inr_excl_gst: Decimal | None
    currency: str | None = None
    exchange_rate: Decimal | None = None
    amount_before_conversion: Decimal | None = None
    quantity_used: int | None = None
    gst_rate: Decimal | None = None
    gst_included: bool = False
    billing_period_quoted: str | None = None
    category_billing_period: str | None = None
    selected_raw_price: Decimal | None = None
    selection_detail: str | None = None
    missing_information_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    steps: tuple[NormalizationStep, ...] = ()
    confidence: float = Field(ge=0, le=1)


class LLMExtractionResponse(BaseModel):
    """Schema requested from an LLM adapter. Validation is mandatory."""

    model_config = ConfigDict(extra="forbid")

    usable_price: bool
    offered_price: Decimal | None = None
    currency: str | None = None
    is_total: bool = False
    total_quantity: int | None = None
    unit: str | None = None
    billing_period: Literal["month", "year"] | None = None
    gst_included: bool | None = None
    discount_percent: Decimal | None = None
    discount_base: Literal["list_price", "previous_quote"] | None = None
    list_price: Decimal | None = None
    is_correction: bool = False
    superseded_price: Decimal | None = None
    free_units: int | None = None
    paid_units: int | None = None
    volume_discount_percent: Decimal | None = None
    volume_threshold: int | None = None
    tier_threshold: int | None = None
    tier_prices: tuple[Decimal, ...] = ()
    ambiguities: tuple[str, ...] = ()
    injection_signals: tuple[str, ...] = ()
    source_text: str
    confidence: float = Field(ge=0, le=1)
