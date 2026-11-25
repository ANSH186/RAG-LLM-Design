"""Resolved, auditable policy snapshot for a single negotiation."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.knowledge.schema import SupplierStatus


class SourceProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    source_file: str
    section_id: str | None = None
    fact: str
    value: str | None = None
    effective_from: date | None = None
    detail: str | None = None


class PolicySnapshot(BaseModel):
    """The exact policy values applicable to one thread on one date."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    policy_version: str
    configuration_hash: str
    thread_id: str
    category_slug: str
    category_title: str
    supplier_id: str
    supplier_name: str
    supplier_status: SupplierStatus
    negotiation_date: date
    normalized_unit: str
    normalized_period: str | None
    target_price: Decimal
    walkaway_price: Decimal
    approval_limit: Decimal
    gst_rate: Decimal
    negotiation_rounds: int
    concession_beta: Decimal
    currency_rates: dict[str, Decimal]
    supplier_exchange_overrides: dict[str, Decimal]
    supplier_price_cap: Decimal | None = None
    amendments_applied: tuple[str, ...] = ()
    source_documents: tuple[str, ...] = ()
    field_provenance: dict[str, tuple[SourceProvenance, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_snapshot(self) -> PolicySnapshot:
        if self.target_price < 0 or self.walkaway_price < 0 or self.approval_limit < 0:
            raise ValueError(f"{self.thread_id}: negative monetary policy value")
        if self.walkaway_price < self.target_price:
            raise ValueError(f"{self.thread_id}: resolved walk-away below target")
        if not (Decimal(0) <= self.gst_rate < Decimal(1)):
            raise ValueError(f"{self.thread_id}: invalid GST rate")
        if self.negotiation_rounds < 2:
            raise ValueError(f"{self.thread_id}: invalid negotiation rounds")
        return self

    def effective_rate(self, currency: str) -> Decimal:
        """Currency rate after supplier contract overrides (never guessed)."""
        code = currency.upper()
        if code == "INR":
            return Decimal(1)
        if code in self.supplier_exchange_overrides:
            return self.supplier_exchange_overrides[code]
        if code in self.currency_rates:
            return self.currency_rates[code]
        raise KeyError(f"no official exchange rate for {code}")
