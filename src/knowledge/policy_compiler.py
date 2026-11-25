"""Compile the Markdown playbook into a validated, structured knowledge base.

The compiler never invents values. Every structured field records the document it
came from, and any parsing ambiguity raises KnowledgeIngestionError.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from src.errors import KnowledgeIngestionError, KnowledgeValidationError
from src.knowledge.parser import (
    category_slug_from_file,
    make_record,
    normalised_unit_parts,
    parse_bullets,
    parse_decimal,
    parse_effective_date,
    parse_int,
    parse_pack_relations,
    parse_percentage,
    parse_rupees,
)
from src.knowledge.schema import (
    Amendment,
    CategoryPolicy,
    CompiledKnowledge,
    CurrencyRate,
    DocumentType,
    ExchangeRateOverride,
    KnowledgeRecord,
    PrecedentRecord,
    PriceCap,
    SupplierContract,
    SupplierStatus,
)
from src.utils.hashing import sha256_json
from src.utils.text import normalise_slug

REQUIRED_CATEGORY_FIELDS = {
    "Normalized unit": "normalized_unit",
    "GST rate": "gst_rate",
    "Target price": "target_price",
    "Walk-away price": "walkaway_price",
    "Negotiation rounds (R)": "negotiation_rounds",
    "Concession exponent (β)": "concession_beta",
    "Approval limit": "approval_limit",
}

REQUIRED_POLICY_KINDS = {
    "decision_procedure": "decision-procedure",
    "banned_terms": "banned-terms",
    "security_policy": "security-policy",
    "units_and_currency": "units-and-currency",
    "approval_limits": "approval-limits",
    "concession_schedule": "concession-schedule",
    "output_format": "output-format",
}


def _title_without_prefix(title: str, prefix: str) -> str:
    match = re.match(rf"(?i)^{re.escape(prefix)}\s*:\s*(.+)$", title)
    return match.group(1).strip() if match else title.strip()


def _category_field_source(record: KnowledgeRecord, key: str) -> str:
    for section in record.sections:
        if key.lower() in section.content.lower():
            return section.section_id
    return record.document_id


def parse_category(
    path: Path,
    root: Path,
    category_titles: dict[str, str] | None = None,
) -> tuple[CategoryPolicy, KnowledgeRecord]:
    record = make_record(path=path, root=root, document_type=DocumentType.CATEGORY.value)
    values = parse_bullets(record.content)
    missing = [key for key in REQUIRED_CATEGORY_FIELDS if key not in values]
    if missing:
        raise KnowledgeIngestionError(f"{path.name}: missing category fields: {missing}")

    title = _title_without_prefix(record.title, "Category")
    slug = category_slug_from_file(path)
    normalized_unit = values["Normalized unit"]
    noun, period = normalised_unit_parts(normalized_unit)
    pack_text, factors = parse_pack_relations(record.content, noun)
    factors.setdefault("dozen", Decimal(12))
    factors.setdefault("unit", Decimal(1))
    factors.setdefault(noun, Decimal(1))

    field_sources = {
        field: _category_field_source(record, key) for key, field in REQUIRED_CATEGORY_FIELDS.items()
    }
    if pack_text:
        field_sources["subunit_factors"] = _category_field_source(record, "Pack conversion")

    category = CategoryPolicy(
        slug=slug,
        title=title,
        document_id=record.document_id,
        source_file=record.source_file,
        normalized_unit=normalized_unit,
        normalized_unit_noun=noun,
        normalized_period=period,
        gst_rate=parse_percentage(values["GST rate"], f"{path.name} GST rate"),
        target_price=parse_rupees(values["Target price"], f"{path.name} target"),
        walkaway_price=parse_rupees(values["Walk-away price"], f"{path.name} walk-away"),
        negotiation_rounds=parse_int(values["Negotiation rounds (R)"], f"{path.name} rounds"),
        concession_beta=parse_decimal(values["Concession exponent (β)"], f"{path.name} beta"),
        approval_limit=parse_rupees(values["Approval limit"], f"{path.name} approval limit"),
        pack_conversion_text=pack_text,
        subunit_factors={k: v for k, v in factors.items()},
        field_sources=field_sources,
    )
    return category, record


def _resolve_category_title(
    title: str, title_to_slug: dict[str, str], context: str
) -> str:
    key = title.strip().lower()
    if key not in title_to_slug:
        raise KnowledgeIngestionError(f"{context}: unknown category title {title!r}")
    return title_to_slug[key]


def parse_supplier(
    path: Path, root: Path, title_to_slug: dict[str, str]
) -> tuple[SupplierContract, KnowledgeRecord]:
    record = make_record(path=path, root=root, document_type=DocumentType.SUPPLIER.value)
    values = parse_bullets(record.content)
    name = _title_without_prefix(record.title, "Supplier")
    if "Status" not in values:
        raise KnowledgeIngestionError(f"{path.name}: missing supplier status")
    status_text = values["Status"]
    upper = status_text.upper()
    if "BLACKLIST" in upper:
        status = SupplierStatus.BLACKLISTED
    elif "PROBATION" in upper:
        status = SupplierStatus.PROBATION
    elif "APPROVED" in upper:
        status = SupplierStatus.APPROVED
    else:
        raise KnowledgeIngestionError(f"{path.name}: unrecognised supplier status {status_text!r}")

    contract = values.get("Contract", "").strip()
    overrides: list[ExchangeRateOverride] = []
    caps: list[PriceCap] = []
    if contract:
        for clause in [part.strip() for part in contract.split(";") if part.strip()]:
            exchange_match = re.search(
                r"invoices?\s+in\s+([A-Z]{3})\s+are\s+converted\s+at\s+(?:₹|Rs\.?|INR)\s*([\d][\d,]*(?:\.\d+)?)\s+per\s+\1",
                clause,
                re.IGNORECASE,
            )
            if exchange_match:
                currency = exchange_match.group(1).upper()
                overrides.append(
                    ExchangeRateOverride(
                        currency=currency,
                        inr_per_unit=parse_rupees(exchange_match.group(2), f"{path.name} contract rate"),
                        source_section=f"{record.document_id}#contract",
                    )
                )
                continue
            cap_match = re.search(
                r"price\s+cap\s+for\s+(.+?)\s+of\s+(?:₹|Rs\.?|INR)\s*([\d][\d,]*(?:\.\d+)?)\s+per\s+([^()]+)",
                clause,
                re.IGNORECASE,
            )
            if cap_match:
                category_title = cap_match.group(1).strip()
                slug = _resolve_category_title(category_title, title_to_slug, path.name)
                caps.append(
                    PriceCap(
                        category_slug=slug,
                        category_title=category_title,
                        value=parse_rupees(cap_match.group(2), f"{path.name} price cap"),
                        unit_text=cap_match.group(3).strip(),
                        source_section=f"{record.document_id}#contract",
                    )
                )
                continue
            raise KnowledgeIngestionError(f"{path.name}: unrecognised contract clause {clause!r}")

    supplier = SupplierContract(
        supplier_id=normalise_slug(name),
        name=name,
        document_id=record.document_id,
        source_file=record.source_file,
        status=status,
        status_text=status_text,
        exchange_rate_overrides=tuple(overrides),
        price_caps=tuple(caps),
        notes=values.get("Notes") or None,
    )
    return supplier, record


def parse_amendment(
    path: Path, root: Path, title_to_slug: dict[str, str]
) -> tuple[Amendment, KnowledgeRecord]:
    record = make_record(path=path, root=root, document_type=DocumentType.AMENDMENT.value)
    effective_from = parse_effective_date(record.content, path.name)
    amendment_id_match = re.search(r"(AMD-\d+)", record.title, re.IGNORECASE)
    amendment_id = amendment_id_match.group(1).upper() if amendment_id_match else path.stem.upper()
    precedence_match = re.search(r"(\d+)$", amendment_id)
    precedence = int(precedence_match.group(1)) if precedence_match else 0
    values = parse_bullets(record.content)
    change = values.get("Change")
    if not change:
        raise KnowledgeIngestionError(f"{path.name}: missing Change field")
    match = re.match(
        r"(?i)^the\s+(walk-away price|walkaway price|target price|approval limit|negotiation rounds|concession exponent|gst rate)\s+for\s+(.+?)\s+is\s+now\s+(.+?)\.?$",
        change,
    )
    if not match:
        raise KnowledgeIngestionError(f"{path.name}: unrecognised change clause {change!r}")
    field_label = match.group(1).strip()
    category_title = match.group(2).strip()
    value_text = match.group(3).strip()
    category_slug = _resolve_category_title(category_title, title_to_slug, path.name)
    field_map = {
        "walk-away price": "walkaway_price",
        "walkaway price": "walkaway_price",
        "target price": "target_price",
        "approval limit": "approval_limit",
        "negotiation rounds": "negotiation_rounds",
        "concession exponent": "concession_beta",
        "gst rate": "gst_rate",
    }
    field = field_map[field_label.lower()]
    if field in {"walkaway_price", "target_price", "approval_limit"}:
        value = parse_rupees(value_text, f"{path.name} change")
    elif field == "negotiation_rounds":
        value = Decimal(parse_int(value_text, f"{path.name} rounds"))
    elif field == "gst_rate":
        value = parse_percentage(value_text, f"{path.name} GST")
    else:
        value = parse_decimal(value_text, f"{path.name} exponent")
    amendment = Amendment(
        amendment_id=amendment_id,
        document_id=record.document_id,
        source_file=record.source_file,
        category_slug=category_slug,
        category_title=category_title,
        effective_from=effective_from,
        field=field,
        field_label=field_label,
        value=value,
        change_text=change,
        precedence=precedence,
    )
    return amendment, record.model_copy(
        update={
            "category": category_slug,
            "amendment_id": amendment_id,
            "effective_from": effective_from,
            "precedence": precedence,
        }
    )


def parse_precedent(
    path: Path, root: Path, title_to_slug: dict[str, str]
) -> tuple[PrecedentRecord, KnowledgeRecord]:
    record = make_record(path=path, root=root, document_type=DocumentType.PRECEDENT.value)
    from src.utils.text import normalise_text

    title = record.title
    category_slug: str | None = None
    match = re.search(r"\s[—–-]\s(.+)$", normalise_text(title))
    if match:
        category_title = match.group(1).strip()
        category_slug = title_to_slug.get(category_title.lower())
        if category_slug is None:
            raise KnowledgeIngestionError(
                f"{path.name}: precedent category {category_title!r} is not a known category"
            )
    precedent_id_match = re.search(r"(P-\d+)", path.stem, re.IGNORECASE)
    precedent_id = precedent_id_match.group(1).upper() if precedent_id_match else path.stem
    precedent = PrecedentRecord(
        precedent_id=precedent_id,
        document_id=record.document_id,
        source_file=record.source_file,
        title=title,
        category_slug=category_slug,
        content=record.content,
    )
    return precedent, record


def parse_currency_table(
    record: KnowledgeRecord,
) -> dict[str, CurrencyRate]:
    rates: dict[str, CurrencyRate] = {}
    for line in record.content.splitlines():
        match = re.match(r"^\|\s*([A-Z]{3})\s*\|\s*([\d.]+)\s*\|", line.strip())
        if not match:
            continue
        currency, raw = match.group(1), match.group(2)
        try:
            rate = Decimal(raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise KnowledgeIngestionError(f"invalid currency rate {raw!r}") from exc
        if rate <= 0:
            raise KnowledgeIngestionError(f"non-positive currency rate for {currency}")
        rates[currency] = CurrencyRate(
            currency=currency,
            inr_per_unit=rate,
            source_document_id=record.document_id,
            source_section=f"{record.document_id}#currency-table",
        )
    if not rates:
        raise KnowledgeIngestionError(
            f"{record.document_id}: no currency table parsed; refusing to continue"
        )
    return rates


def compile_knowledge(
    *,
    knowledge_root: str | Path,
    environment: str,
) -> CompiledKnowledge:
    root = Path(knowledge_root)
    if not root.exists():
        raise KnowledgeIngestionError(f"knowledge root does not exist: {root}")
    documents: list[KnowledgeRecord] = []
    categories: dict[str, CategoryPolicy] = {}
    title_to_slug: dict[str, str] = {}

    category_paths = sorted((root / "categories").glob("*.md"))
    if not category_paths:
        raise KnowledgeIngestionError(f"{root}: no category documents found")
    for path in category_paths:
        category, record = parse_category(path, root)
        if category.slug in categories:
            raise KnowledgeIngestionError(f"duplicate category slug: {category.slug}")
        categories[category.slug] = category
        title_to_slug[category.title.lower()] = category.slug
        documents.append(record.model_copy(update={"category": category.slug}))

    suppliers: dict[str, SupplierContract] = {}
    supplier_paths = sorted((root / "suppliers").glob("*.md"))
    if not supplier_paths:
        raise KnowledgeIngestionError(f"{root}: no supplier documents found")
    for path in supplier_paths:
        supplier, record = parse_supplier(path, root, title_to_slug)
        if supplier.supplier_id in suppliers:
            raise KnowledgeIngestionError(f"duplicate supplier id: {supplier.supplier_id}")
        suppliers[supplier.supplier_id] = supplier
        documents.append(record.model_copy(update={"supplier": supplier.supplier_id}))

    amendments: list[Amendment] = []
    for path in sorted((root / "amendments").glob("*.md")):
        amendment, record = parse_amendment(path, root, title_to_slug)
        amendments.append(amendment)
        documents.append(record)
    amendments.sort(key=lambda a: (a.effective_from, a.precedence, a.amendment_id))

    policy_documents: dict[str, str] = {}
    currency_rates: dict[str, CurrencyRate] = {}
    canonical_policy_keys = {stem: canonical for canonical, stem in REQUIRED_POLICY_KINDS.items()}
    policy_paths = sorted((root / "policies").glob("*.md"))
    for path in policy_paths:
        kind = path.stem.lower()
        record = make_record(path=path, root=root, document_type=DocumentType.POLICY.value)
        documents.append(record)
        canonical = canonical_policy_keys.get(kind, kind)
        policy_documents[canonical] = record.document_id
        if kind == "units-and-currency":
            currency_rates = parse_currency_table(record)
    readme_path = root / "README.md"
    if readme_path.exists():
        readme = make_record(path=readme_path, root=root, document_type=DocumentType.POLICY.value)
        documents.append(readme)
        policy_documents["readme"] = readme.document_id

    missing_policies = [kind for kind in REQUIRED_POLICY_KINDS if kind not in policy_documents]
    if missing_policies:
        raise KnowledgeIngestionError(f"missing mandatory policy documents: {missing_policies}")
    if not currency_rates:
        raise KnowledgeIngestionError("units-and-currency.md did not compile any rates")

    precedents: list[PrecedentRecord] = []
    for path in sorted((root / "precedents").glob("*.md")):
        precedent, record = parse_precedent(path, root, title_to_slug)
        precedents.append(precedent)
        documents.append(record.model_copy(update={"category": precedent.category_slug}))

    # Fail loudly on duplicate document IDs.
    seen_ids: set[str] = set()
    for record in documents:
        if record.document_id in seen_ids:
            raise KnowledgeIngestionError(f"duplicate document id: {record.document_id}")
        seen_ids.add(record.document_id)

    payload = {
        "environment": environment,
        "categories": {k: v.model_dump(mode="json") for k, v in categories.items()},
        "suppliers": {k: v.model_dump(mode="json") for k, v in suppliers.items()},
        "amendments": [a.model_dump(mode="json") for a in amendments],
        "currency_rates": {k: v.model_dump(mode="json") for k, v in currency_rates.items()},
        "precedents": [p.model_dump(mode="json") for p in precedents],
        "policy_documents": policy_documents,
    }
    return CompiledKnowledge(
        environment=environment,
        source_root=str(root).replace("\\", "/"),
        compiled_at=datetime.now(UTC).isoformat(),
        categories=categories,
        suppliers=suppliers,
        amendments=tuple(amendments),
        currency_rates=currency_rates,
        precedents=tuple(precedents),
        documents=tuple(documents),
        policy_document_ids=policy_documents,
        configuration_hash=sha256_json(payload),
    )


def validate_compiled_knowledge(knowledge: CompiledKnowledge) -> None:
    """Structural validation after compilation. Raises on any official-rule gap."""
    if not knowledge.categories:
        raise KnowledgeValidationError("no categories")
    if not knowledge.suppliers:
        raise KnowledgeValidationError("no suppliers")
    for slug, category in knowledge.categories.items():
        if category.document_id not in {doc.document_id for doc in knowledge.documents}:
            raise KnowledgeValidationError(f"{slug}: category source document missing")
        for field in (
            "target_price",
            "walkaway_price",
            "approval_limit",
            "gst_rate",
            "negotiation_rounds",
            "concession_beta",
        ):
            if getattr(category, field) is None:
                raise KnowledgeValidationError(f"{slug}: missing {field}")
        if category.normalized_period not in {None, "month", "year"}:
            raise KnowledgeValidationError(f"{slug}: invalid normalized period")
    for supplier_id, supplier in knowledge.suppliers.items():
        if supplier.status not in {
            SupplierStatus.APPROVED,
            SupplierStatus.PROBATION,
            SupplierStatus.BLACKLISTED,
        }:
            raise KnowledgeValidationError(f"{supplier_id}: invalid status")
        for cap in supplier.price_caps:
            if cap.category_slug not in knowledge.categories:
                raise KnowledgeValidationError(f"{supplier_id}: cap category unknown")
    for amendment in knowledge.amendments:
        if amendment.category_slug not in knowledge.categories:
            raise KnowledgeValidationError(f"{amendment.amendment_id}: unknown category")
        if amendment.effective_from.year < 2000:
            raise KnowledgeValidationError(f"{amendment.amendment_id}: implausible date")
    for policy in REQUIRED_POLICY_KINDS:
        if policy not in knowledge.policy_document_ids:
            raise KnowledgeValidationError(f"mandatory policy {policy} missing")
    for currency in ("USD", "EUR", "GBP"):
        # These are the currencies explicitly stated by the Acme playbook. A different
        # environment may differ, so the engine itself remains table-driven.
        if currency not in knowledge.currency_rates and knowledge.environment == "acme":
            raise KnowledgeValidationError(f"missing official rate for {currency}")
