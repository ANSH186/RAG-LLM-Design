"""Markdown parsing for the procurement playbook.

Parsing is deliberately strict. If an official document cannot be interpreted, the
ingestion run fails rather than silently producing incomplete procurement knowledge.
"""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.errors import KnowledgeIngestionError
from src.knowledge.schema import (
    KnowledgeRecord,
    KnowledgeSection,
)
from src.utils.hashing import sha256_text
from src.utils.numbers import parse_decimal_number
from src.utils.text import normalise_slug, split_sentences

BULLET_RE = re.compile(r"^\s*-\s+\*\*(?P<key>[^*]+?):\*\*\s*(?P<value>.+?)\s*$")
HEADING_RE = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
RUPEE_RE = re.compile(r"(?:₹|Rs\.?|INR)\s*([\d][\d,]*(?:\.\d+)?)", re.IGNORECASE)


def read_markdown(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise KnowledgeIngestionError(f"{path}: not valid UTF-8: {exc}") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise KnowledgeIngestionError(f"{path}: empty document")
    return text


def parse_bullets(text: str) -> dict[str, str]:
    """Parse '**Key:** value' bullets, tolerating indentation used in the source."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = BULLET_RE.match(line)
        if not match:
            continue
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key in values:
            raise KnowledgeIngestionError(f"duplicate field {key!r}")
        values[key] = value
    return values


def parse_rupees(value: str, context: str) -> Decimal:
    match = RUPEE_RE.search(value)
    if not match:
        # Callers sometimes provide the already-captured number (e.g. from a contract clause).
        # Accept a bare numeric token, but never guess an amount from arbitrary text.
        bare = re.fullmatch(r"\s*([\d][\d,]*(?:\.\d+)?)\s*", value)
        if bare:
            raw = bare.group(1)
            try:
                return parse_decimal_number(raw)
            except InvalidOperation as exc:
                raise KnowledgeIngestionError(f"{context}: invalid rupee amount {raw!r}") from exc
        raise KnowledgeIngestionError(f"{context}: no rupee amount in {value!r}")
    raw = match.group(1)
    try:
        return parse_decimal_number(raw)
    except InvalidOperation as exc:
        raise KnowledgeIngestionError(f"{context}: invalid rupee amount {raw!r}") from exc


def parse_percentage(value: str, context: str) -> Decimal:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", value)
    if not match:
        raise KnowledgeIngestionError(f"{context}: no percentage in {value!r}")
    percent = parse_decimal_number(match.group(1))
    return percent / Decimal(100)


def parse_int(value: str, context: str) -> int:
    match = re.search(r"\d+", value)
    if not match:
        raise KnowledgeIngestionError(f"{context}: no integer in {value!r}")
    return int(match.group(0))


def parse_decimal(value: str, context: str) -> Decimal:
    match = re.search(r"\d+(?:\.\d+)?", value)
    if not match:
        raise KnowledgeIngestionError(f"{context}: no decimal in {value!r}")
    return Decimal(match.group(0))


def classify_document(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if not relative.parts:
        raise KnowledgeIngestionError(f"{path}: cannot classify document")
    folder = relative.parts[0].lower()
    folder = folder.removesuffix("s")
    mapping = {
        "policie": "policy",
        "policy": "policy",
        "categorie": "category",
        "category": "category",
        "supplier": "supplier",
        "amendment": "amendment",
        "precedent": "precedent",
    }
    # 'policies' -> 'policie' after naive singularisation; handle explicitly.
    if relative.parts[0].lower() == "policies":
        return "policy"
    if folder not in mapping:
        raise KnowledgeIngestionError(f"{path}: unknown playbook folder {relative.parts[0]!r}")
    return mapping[folder]


def sectionise(text: str, document_id: str, max_chars: int = 1200) -> tuple[KnowledgeSection, ...]:
    """Split a document into provenance-preserving heading sections.

    Headings are the primary boundaries. Very long sections are split at blank-line
    paragraph boundaries, but each resulting chunk retains the heading and verbatim
    source text. Small playbook documents remain readable as a single evidence unit.
    """
    matches = list(HEADING_RE.finditer(text))
    ranges: list[tuple[str | None, str]] = []
    if not matches:
        ranges.append((None, text.strip()))
    else:
        if matches[0].start() > 0:
            ranges.append((None, text[: matches[0].start()].strip()))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[match.start() : end].strip()
            ranges.append((match.group("title").strip(), body))
    sections: list[KnowledgeSection] = []
    ordinal = 0
    for heading, body in ranges:
        if not body:
            continue
        subchunks = [body]
        if len(body) > max_chars:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            subchunks = []
            buffer: list[str] = []
            for paragraph in paragraphs:
                candidate = "\n\n".join(buffer + [paragraph])
                if buffer and len(candidate) > max_chars:
                    subchunks.append("\n\n".join(buffer))
                    buffer = [paragraph]
                else:
                    buffer.append(paragraph)
            if buffer:
                subchunks.append("\n\n".join(buffer))
        for sub in subchunks:
            sections.append(
                KnowledgeSection(
                    section_id=f"{document_id}#s{ordinal}",
                    heading=heading or document_id,
                    content=sub.strip(),
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    if not sections:
        sections.append(
            KnowledgeSection(
                section_id=f"{document_id}#s0",
                heading=document_id,
                content=text.strip(),
                ordinal=0,
            )
        )
    return tuple(sections)


def extract_title(text: str, fallback: str) -> str:
    match = HEADING_RE.search(text)
    if match:
        return match.group("title").strip()
    return fallback


def parse_effective_date(text: str, context: str) -> date:
    match = re.search(r"Effective from:\*\*\s*(\d{4}-\d{2}-\d{2})", text)
    if not match:
        raise KnowledgeIngestionError(f"{context}: missing Effective from date")
    try:
        return date.fromisoformat(match.group(1))
    except ValueError as exc:
        raise KnowledgeIngestionError(f"{context}: invalid effective date") from exc


def normalised_unit_parts(normalized_unit: str) -> tuple[str, str | None]:
    """Return (noun, billing period) from e.g. 'INR excl. GST per site per month'."""
    unit = re.sub(r"(?i)^.*?\bper\b\s*", "", normalized_unit.strip())
    period_match = re.search(r"(?i)\s+per\s+(month|year|annum)\s*$", unit)
    period: str | None = None
    if period_match:
        period = "year" if period_match.group(1).lower() in {"year", "annum"} else "month"
        unit = unit[: period_match.start()]
    noun = unit.strip().lower()
    noun = re.sub(r"^(?:a|an|one)\s+", "", noun)
    return noun, period


def parse_pack_relations(
    text: str, normalized_noun: str
) -> tuple[str | None, dict[str, Decimal]]:
    """Compile pack/sub-unit conversions from a category page.

    Returns (pack conversion source sentence, mapping quoted_unit -> normalized units
    per quoted unit). Example: 'one box = 10 reams' with normalized noun 'box'
    yields {'ream': Decimal('0.1')}.
    """
    factors: dict[str, Decimal] = {}
    source: str | None = None
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if "pack" in lowered and "hold" in lowered:
            match = re.search(r"(?:a\s+)?pack\s+holds\s+(\d+)\s+([a-z ]+)", lowered)
            if match:
                factors["pack"] = Decimal(match.group(1))
                source = sentence
                continue
        match = re.search(
            r"(?:one|1|a|an)\s+([a-z]+)\s*=\s*(\d+)\s+([a-z]+)", lowered
        )
        if match:
            left, count, right = match.group(1), Decimal(match.group(2)), match.group(3)
            right = right.rstrip("s")
            if left.rstrip("s") == normalized_noun.rstrip("s"):
                # one normalized unit contains `count` quoted sub-units
                factors[right] = Decimal(1) / count
                source = sentence
            elif right == normalized_noun.rstrip("s"):
                # one quoted container contains `count` normalized units
                factors[left.rstrip("s")] = count
                source = sentence
    return source, factors


def make_record(
    *,
    path: Path,
    root: Path,
    document_type: str,
    title_override: str | None = None,
    category: str | None = None,
    supplier: str | None = None,
    effective_from: date | None = None,
    amendment_id: str | None = None,
    precedence: int = 0,
    metadata: dict | None = None,
) -> KnowledgeRecord:
    text = read_markdown(path)
    document_id = path.stem
    title = title_override or extract_title(text, document_id)
    return KnowledgeRecord(
        document_id=document_id,
        document_type=document_type,  # type: ignore[arg-type]
        source_file=str(path.relative_to(root)).replace("\\", "/"),
        title=title,
        content=text,
        content_hash=sha256_text(text),
        category=category,
        supplier=supplier,
        effective_from=effective_from,
        amendment_id=amendment_id,
        precedence=precedence,
        metadata=metadata or {},
        sections=sectionise(text, document_id),
    )


def category_slug_from_file(path: Path) -> str:
    return normalise_slug(path.stem)


def supplier_slug_from_title(title: str) -> str:
    return normalise_slug(title)
