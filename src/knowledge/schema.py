"""Validated knowledge models for the procurement playbook."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentType(str, Enum):
    POLICY = "policy"
    CATEGORY = "category"
    SUPPLIER = "supplier"
    AMENDMENT = "amendment"
    PRECEDENT = "precedent"


class SupplierStatus(str, Enum):
    APPROVED = "APPROVED"
    PROBATION = "PROBATION"
    BLACKLISTED = "BLACKLISTED"


class KnowledgeSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    section_id: str
    heading: str
    content: str
    ordinal: int = Field(ge=0)


class KnowledgeRecord(BaseModel):
    """One source Markdown document, faithfully parsed and sectioned."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_type: DocumentType
    source_file: str
    title: str
    content: str
    content_hash: str
    category: str | None = None
    supplier: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    amendment_id: str | None = None
    precedence: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: tuple[KnowledgeSection, ...] = ()

    @model_validator(mode="after")
    def validate_temporal(self) -> KnowledgeRecord:
        if self.effective_from and self.effective_to and self.effective_to < self.effective_from:
            raise ValueError(f"{self.document_id}: effective_to precedes effective_from")
        return self


class CategoryPolicy(BaseModel):
    """Structured category values compiled from a category page.

    The Markdown source remains the authority; this model is a validated projection.
    """

    model_config = ConfigDict(frozen=True)

    slug: str
    title: str
    document_id: str
    source_file: str
    normalized_unit: str
    normalized_unit_noun: str
    normalized_period: str | None = None
    gst_rate: Decimal
    target_price: Decimal
    walkaway_price: Decimal
    negotiation_rounds: int = Field(ge=2)
    concession_beta: Decimal = Field(gt=0)
    approval_limit: Decimal
    pack_conversion_text: str | None = None
    subunit_factors: dict[str, Decimal] = Field(default_factory=dict)
    field_sources: dict[str, str] = Field(default_factory=dict)

    @field_validator("gst_rate")
    @classmethod
    def gst_between_zero_and_one(cls, value: Decimal) -> Decimal:
        if not (Decimal(0) <= value < Decimal(1)):
            raise ValueError(f"GST rate must be a fraction below 1: {value}")
        return value

    @model_validator(mode="after")
    def validate_category_values(self) -> CategoryPolicy:
        if self.target_price < 0 or self.walkaway_price < 0 or self.approval_limit < 0:
            raise ValueError(f"{self.slug}: negative monetary value")
        if self.walkaway_price < self.target_price:
            raise ValueError(f"{self.slug}: walkaway below target")
        return self


class ExchangeRateOverride(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str
    inr_per_unit: Decimal = Field(gt=0)
    source_section: str


class PriceCap(BaseModel):
    model_config = ConfigDict(frozen=True)

    category_slug: str
    category_title: str
    value: Decimal = Field(gt=0)
    unit_text: str
    source_section: str


class SupplierContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    supplier_id: str
    name: str
    document_id: str
    source_file: str
    status: SupplierStatus
    status_text: str
    exchange_rate_overrides: tuple[ExchangeRateOverride, ...] = ()
    price_caps: tuple[PriceCap, ...] = ()
    notes: str | None = None


class Amendment(BaseModel):
    model_config = ConfigDict(frozen=True)

    amendment_id: str
    document_id: str
    source_file: str
    category_slug: str
    category_title: str
    effective_from: date
    field: str
    field_label: str
    value: Decimal
    change_text: str
    precedence: int = 0


class CurrencyRate(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str
    inr_per_unit: Decimal = Field(gt=0)
    source_document_id: str
    source_section: str


class PrecedentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    precedent_id: str
    document_id: str
    source_file: str
    title: str
    category_slug: str | None
    content: str


class CompiledKnowledge(BaseModel):
    """The complete validated knowledge base for one procurement environment."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    environment: str
    source_root: str
    compiled_at: str
    categories: dict[str, CategoryPolicy]
    suppliers: dict[str, SupplierContract]
    amendments: tuple[Amendment, ...]
    currency_rates: dict[str, CurrencyRate]
    precedents: tuple[PrecedentRecord, ...]
    documents: tuple[KnowledgeRecord, ...]
    policy_document_ids: dict[str, str]
    configuration_hash: str

    @model_validator(mode="after")
    def validate_cross_references(self) -> CompiledKnowledge:
        if not self.categories:
            raise ValueError("no categories compiled")
        if not self.suppliers:
            raise ValueError("no suppliers compiled")
        if not self.currency_rates:
            raise ValueError("no currency rates compiled")
        for amendment in self.amendments:
            if amendment.category_slug not in self.categories:
                raise ValueError(f"amendment {amendment.amendment_id} references unknown category")
        for supplier in self.suppliers.values():
            for cap in supplier.price_caps:
                if cap.category_slug not in self.categories:
                    raise ValueError(
                        f"supplier {supplier.supplier_id} cap references unknown category {cap.category_slug}"
                    )
        return self

    def supplier_by_name(self, name: str) -> SupplierContract:
        key = name.strip().lower()
        for supplier in self.suppliers.values():
            if supplier.name.strip().lower() == key:
                return supplier
        raise KeyError(f"unknown supplier: {name}")

    def category_by_slug(self, slug: str) -> CategoryPolicy:
        try:
            return self.categories[slug]
        except KeyError as exc:
            raise KeyError(f"unknown category: {slug}") from exc
