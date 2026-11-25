"""Temporal policy resolution.

Given (category, supplier, negotiation date), resolve the base category page, apply all
amendments effective on or before that date in effective-date order, then apply the
supplier contract overrides. Future amendments are never visible to earlier threads.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from src.errors import TemporalResolutionError
from src.knowledge.policy_snapshot import PolicySnapshot, SourceProvenance
from src.knowledge.schema import CompiledKnowledge
from src.utils.hashing import sha256_json
from src.utils.time import parse_iso_date

SNAPSHOT_FIELDS = (
    "target_price",
    "walkaway_price",
    "approval_limit",
    "gst_rate",
    "negotiation_rounds",
    "concession_beta",
)


def _doc_section_source(
    knowledge: CompiledKnowledge,
    document_id: str,
    section_id: str | None,
    fact: str,
    value: object,
    detail: str | None = None,
) -> SourceProvenance:
    doc = next((d for d in knowledge.documents if d.document_id == document_id), None)
    if doc is None:
        raise TemporalResolutionError(f"source document {document_id} is missing")
    return SourceProvenance(
        document_id=document_id,
        source_file=doc.source_file,
        section_id=section_id,
        fact=fact,
        value=str(value),
        effective_from=doc.effective_from,
        detail=detail,
    )


def resolve_policy_snapshot(
    knowledge: CompiledKnowledge,
    *,
    thread_id: str,
    category_slug: str,
    supplier_name: str,
    negotiation_date: str | date,
    require_known_supplier: bool = True,
) -> PolicySnapshot:
    day = parse_iso_date(negotiation_date) if isinstance(negotiation_date, str) else negotiation_date
    if category_slug not in knowledge.categories:
        raise TemporalResolutionError(f"{thread_id}: unknown category {category_slug!r}")
    try:
        supplier = knowledge.supplier_by_name(supplier_name)
    except KeyError as exc:
        if require_known_supplier:
            raise TemporalResolutionError(f"{thread_id}: {exc}") from exc
        raise

    category = knowledge.categories[category_slug]
    values: dict[str, Decimal | int] = {
        "target_price": category.target_price,
        "walkaway_price": category.walkaway_price,
        "approval_limit": category.approval_limit,
        "gst_rate": category.gst_rate,
        "negotiation_rounds": category.negotiation_rounds,
        "concession_beta": category.concession_beta,
    }
    provenance: dict[str, list[SourceProvenance]] = {field: [] for field in SNAPSHOT_FIELDS}
    for field in SNAPSHOT_FIELDS:
        section_id = category.field_sources.get(field)
        provenance[field].append(
            _doc_section_source(
                knowledge,
                category.document_id,
                section_id,
                field,
                values[field],
                detail=f"base category value for {category.title}",
            )
        )

    source_documents = {category.document_id, supplier.document_id}
    applied: list[str] = []
    for amendment in knowledge.amendments:
        if amendment.category_slug != category_slug:
            continue
        if amendment.effective_from > day:
            continue
        if amendment.field not in values:
            raise TemporalResolutionError(
                f"{thread_id}: amendment {amendment.amendment_id} changes unsupported field {amendment.field}"
            )
        values[amendment.field] = (
            int(amendment.value) if amendment.field == "negotiation_rounds" else amendment.value
        )
        provenance[amendment.field].append(
            SourceProvenance(
                document_id=amendment.document_id,
                source_file=amendment.source_file,
                section_id=None,
                fact=amendment.field,
                value=str(amendment.value),
                effective_from=amendment.effective_from,
                detail=f"{amendment.amendment_id}: {amendment.change_text}",
            )
        )
        applied.append(amendment.amendment_id)
        source_documents.add(amendment.document_id)

    supplier_cap: Decimal | None = None
    for cap in supplier.price_caps:
        if cap.category_slug != category_slug:
            continue
        if supplier_cap is not None:
            raise TemporalResolutionError(
                f"{thread_id}: multiple supplier price caps for {category_slug}"
            )
        supplier_cap = cap.value
        values["walkaway_price"] = cap.value
        provenance["walkaway_price"].append(
            SourceProvenance(
                document_id=supplier.document_id,
                source_file=supplier.source_file,
                section_id=cap.source_section,
                fact="walkaway_price",
                value=str(cap.value),
                detail=f"supplier contract cap for {category.title} replaces walk-away",
            )
        )

    exchange_overrides = {
        override.currency: override.inr_per_unit for override in supplier.exchange_rate_overrides
    }
    for override in supplier.exchange_rate_overrides:
        provenance.setdefault("currency_rates", []).append(
            SourceProvenance(
                document_id=supplier.document_id,
                source_file=supplier.source_file,
                section_id=override.source_section,
                fact="exchange_rate",
                value=str(override.inr_per_unit),
                detail=f"supplier contract rate for {override.currency}",
            )
        )
    provenance.setdefault("supplier_status", []).append(
        SourceProvenance(
            document_id=supplier.document_id,
            source_file=supplier.source_file,
            section_id=None,
            fact="supplier_status",
            value=supplier.status.value,
            detail=supplier.status_text,
        )
    )
    base_currency_doc = knowledge.policy_document_ids.get("units_and_currency")
    if base_currency_doc:
        base_doc = next((d for d in knowledge.documents if d.document_id == base_currency_doc), None)
        provenance.setdefault("currency_rates", []).append(
            SourceProvenance(
                document_id=base_currency_doc,
                source_file=base_doc.source_file if base_doc else "",
                section_id=f"{base_currency_doc}#currency-table",
                fact="currency_rates",
                value=",".join(sorted(knowledge.currency_rates)),
                detail="official currency conversion table",
            )
        )

    currency_rates = {code: rate.inr_per_unit for code, rate in knowledge.currency_rates.items()}
    payload = {
        "configuration_hash": knowledge.configuration_hash,
        "thread_id": thread_id,
        "category": category_slug,
        "supplier_id": supplier.supplier_id,
        "date": day.isoformat(),
        "values": {k: str(v) for k, v in values.items()},
        "amendments": applied,
        "cap": str(supplier_cap) if supplier_cap is not None else None,
        "overrides": {k: str(v) for k, v in exchange_overrides.items()},
    }
    version_hash = sha256_json(payload)
    policy_version = f"{knowledge.environment}:{day.isoformat()}:{version_hash[:12]}"
    snapshot_id = f"snap-{thread_id}-{version_hash[:12]}"

    return PolicySnapshot(
        snapshot_id=snapshot_id,
        policy_version=policy_version,
        configuration_hash=knowledge.configuration_hash,
        thread_id=thread_id,
        category_slug=category.slug,
        category_title=category.title,
        supplier_id=supplier.supplier_id,
        supplier_name=supplier.name,
        supplier_status=supplier.status,
        negotiation_date=day,
        normalized_unit=category.normalized_unit,
        normalized_period=category.normalized_period,
        target_price=Decimal(values["target_price"]),
        walkaway_price=Decimal(values["walkaway_price"]),
        approval_limit=Decimal(values["approval_limit"]),
        gst_rate=Decimal(values["gst_rate"]),
        negotiation_rounds=int(values["negotiation_rounds"]),
        concession_beta=Decimal(values["concession_beta"]),
        currency_rates=currency_rates,
        supplier_exchange_overrides=exchange_overrides,
        supplier_price_cap=supplier_cap,
        amendments_applied=tuple(applied),
        source_documents=tuple(sorted(source_documents)),
        field_provenance={k: tuple(v) for k, v in provenance.items()},
    )
