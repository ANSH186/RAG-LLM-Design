"""Deterministic supplier-offer extraction.

This parser is the default competition extractor. It is deliberately independent of
any LLM and is paired with an optional, schema-validated LLM extractor for genuinely
unresolved language. It covers all offer constructions documented in the playbook:

* direct unit prices, totals, ranges, tiers, volume discounts, free units
* list-price and previous-quote discounts, corrections
* packs, dozens, reams and per-N lot prices
* GST inclusion, currencies, monthly/yearly subscriptions
* numbers with Indian/Western grouping, lakh/k suffixes and English words
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from src.extraction.extraction_schema import (
    ExtractionMethod,
    OfferExtraction,
    PriceCandidate,
    ValidationStatus,
)
from src.knowledge.policy_snapshot import PolicySnapshot
from src.utils.numbers import parse_decimal_number, words_to_decimal
from src.utils.text import (
    latest_supplier_block,
    split_message_blocks,
)

CURRENCY_RE = (
    r"(?P<cur>₹|Rs\.?|INR|USD|EUR|GBP|JPY|AED|SGD|\$|€|£)"
)
DIGIT_MONEY_RE = re.compile(
    rf"{CURRENCY_RE}\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<suffix>lakhs?|lacs?|crores?|cr|thousand|k)?",
    re.IGNORECASE,
)
WORD_NUMBER = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|lakh|lakhs|"
    r"lac|lacs|million|crore|crores|billion|and)"
)
WORD_MONEY_RE = re.compile(
    rf"(?P<words>\b{WORD_NUMBER}(?:[\s-]+{WORD_NUMBER})*?\s+rupees?)\b",
    re.IGNORECASE,
)

CURRENCY_CODES = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "rupee": "INR",
    "rupees": "INR",
    "usd": "USD",
    "$": "USD",
    "eur": "EUR",
    "€": "EUR",
    "gbp": "GBP",
    "£": "GBP",
    "jpy": "JPY",
    "aed": "AED",
    "sgd": "SGD",
}


@dataclass(frozen=True)
class ParsedPreviousQuote:
    candidates: tuple[PriceCandidate, ...]
    unit_factor: Decimal
    billing_period: str | None
    gst_treatment: str
    is_total: bool
    total_quantity: int | None


def normalise_currency(token: str | None) -> str | None:
    if not token:
        return None
    key = token.strip().lower()
    return CURRENCY_CODES.get(key, key.upper())


def find_money(text: str) -> list[PriceCandidate]:
    """Return non-overlapping amount candidates in source order."""
    matches: list[PriceCandidate] = []
    occupied: list[tuple[int, int]] = []
    for match in DIGIT_MONEY_RE.finditer(text):
        start, end = match.start(), match.end()
        if any(not (end <= a or start >= b) for a, b in occupied):
            continue
        raw_number = match.group("num")
        suffix = match.group("suffix")
        try:
            amount = parse_decimal_number(raw_number, suffix)
        except InvalidOperation:
            continue
        matches.append(
            PriceCandidate(
                amount=amount,
                currency=normalise_currency(match.group("cur")),
                raw_text=match.group(0).strip(),
                start=start,
                end=end,
                numeric_form="digits",
            )
        )
        occupied.append((start, end))
    for match in WORD_MONEY_RE.finditer(text):
        start, end = match.start(), match.end()
        if any(not (end <= a or start >= b) for a, b in occupied):
            continue
        phrase = match.group("words")
        words = re.sub(r"(?i)\s+rupees?$", "", phrase).strip()
        try:
            amount = words_to_decimal(words)
        except ValueError:
            continue
        matches.append(
            PriceCandidate(
                amount=amount,
                currency="INR",
                raw_text=phrase.strip(),
                start=start,
                end=end,
                numeric_form="words",
            )
        )
        occupied.append((start, end))
    matches.sort(key=lambda item: item.start)
    return matches


def detect_billing_period(text: str) -> str | None:
    month = bool(re.search(r"(?i)\bper\s+month\b|\bmonthly\b", text))
    year = bool(re.search(r"(?i)\bper\s+(?:year|annum)\b|\bannual(?:ly)?\b|\bper\s+yr\b", text))
    if month and year:
        return None
    if month:
        return "month"
    if year:
        return "year"
    return None


def detect_gst(text: str) -> str:
    included = bool(
        re.search(
            r"(?i)\b(?:inclusive\s+of|including|incl\.?|inclusive)\s+gst\b|\bgst\s+included\b|\bgst\s+inclusive\b|\bgst-inclusive\b",
            text,
        )
    )
    excluded = bool(
        re.search(r"(?i)\b(?:exclusive\s+of|excluding|excl\.?)\s+gst\b|\bgst\s+extra\b|\bplus\s+gst\b|\bgst\s+excluded\b", text)
    )
    if included and excluded:
        return "ambiguous"
    if included:
        return "included"
    if excluded:
        return "excluded"
    return "unspecified"


def detect_unit_factor(text: str) -> tuple[Decimal, str | None, str | None]:
    """Return (normalised units per quoted unit, quoted label, warning)."""
    lowered = text.lower()
    pack = re.search(r"\bper\s+pack\s+of\s+(\d+)\b", lowered)
    if pack:
        count = Decimal(pack.group(1))
        if count <= 0:
            return Decimal(1), None, "non_positive_pack_size"
        return count, f"pack of {pack.group(1)}", None
    if re.search(r"\bper\s+dozen\b", lowered):
        return Decimal(12), "dozen", None
    per_n = re.search(r"\bper\s+(\d+)\s+([a-z]+)\b", lowered)
    if per_n:
        count = Decimal(per_n.group(1))
        if count <= 0:
            return Decimal(1), None, "non_positive_lot_size"
        return count, f"{per_n.group(1)} {per_n.group(2)}", None
    if re.search(r"\bper\s+ream\b", lowered):
        # Category pages define the relationship (e.g. one box = 10 reams).
        # The caller resolves this through the category configuration.
        return Decimal(1), "ream", "ream_factor_requires_category"
    generic = re.search(r"\bper\s+([a-z]+)\b", lowered)
    if generic and generic.group(1) not in {"month", "year", "annum", "yr"}:
        return Decimal(1), generic.group(1), None
    return Decimal(1), None, None


def detect_terms(latest_text: str) -> dict[str, str | None]:
    terms: dict[str, str | None] = {"payment": None, "delivery": None, "warranty": None}
    for line in latest_text.splitlines():
        lower = line.lower().strip()
        if not line.strip():
            continue
        if terms["payment"] is None and any(
            token in lower for token in ("payment terms", "advance", "upfront", "net-", "net 30", "net 45", "net 60")
        ) and "order value" not in lower:
            terms["payment"] = line.strip()
        if terms["delivery"] is None and "delivery" in lower:
            terms["delivery"] = line.strip()
        if terms["warranty"] is None and "warranty" in lower:
            terms["warranty"] = line.strip()
    return terms


def _select_offer_line(latest_text: str) -> str | None:
    lines = [line.strip() for line in latest_text.splitlines() if line.strip()]
    money_lines = [
        line
        for line in lines
        if find_money(line) or "off our previous quote" in line.lower()
    ]
    if not money_lines:
        return None

    def priority(line: str) -> int:
        lower = line.lower()
        if "correction" in lower or "should read" in lower:
            return 100
        if "in total" in lower or "total for all" in lower:
            return 95
        if "between" in lower:
            return 90
        if "up to" in lower and "larger order" in lower:
            return 85
        if "list price" in lower:
            return 80
        if "previous quote" in lower:
            return 75
        if "you requested" in lower:
            return 70
        if any(phrase in lower for phrase in ("our quote is", "we can offer", "our final price is", "our revised offer is", "we can do")):
            return 60
        return 10

    return min(money_lines, key=lambda line: (-priority(line), lines.index(line)))


def parse_previous_quote(
    message: str, snapshot: PolicySnapshot | None, category_factor_resolver
) -> tuple[str | None, ParsedPreviousQuote | None]:
    supplier_blocks = [b for b in split_message_blocks(message) if b.speaker.lower() != "acme procurement"]
    if len(supplier_blocks) < 2:
        return None, None
    text = supplier_blocks[-2].text.strip()
    candidates = tuple(find_money(text))
    if not candidates:
        return text, None
    factor, label, _warning = detect_unit_factor(text)
    if label and label != "ream":
        # Category pages may define a sub-unit conversion not spelled out in the message.
        base_label = label.split()[0] if label.startswith("pack of") else label
        if factor == 1:
            resolved = category_factor_resolver(base_label)
            if resolved is not None:
                factor = resolved
    if label == "ream":
        resolved = category_factor_resolver("ream")
        if resolved is None:
            return text, None
        factor = resolved
    return text, ParsedPreviousQuote(
        candidates=candidates,
        unit_factor=factor,
        billing_period=detect_billing_period(text),
        gst_treatment=detect_gst(text),
        is_total=bool(re.search(r"(?i)\bin total\b|total for all", text)),
        total_quantity=_parse_total_quantity(text),
    )


def _parse_total_quantity(text: str) -> int | None:
    match = re.search(r"(?i)\bfor\s+all\s+([\d,]+)", text)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw.replace(",", ""))
    except ValueError:
        return None


class DeterministicExtractor:
    def __init__(self, *, category_factor_resolver) -> None:
        self.category_factor_resolver = category_factor_resolver

    def extract(
        self,
        *,
        thread_id: str,
        source_message: str,
        snapshot: PolicySnapshot,
    ) -> OfferExtraction:
        block = latest_supplier_block(source_message)
        latest_text = block.text
        offer_line = _select_offer_line(latest_text)
        ambiguities: list[str] = []
        preliminary_missing: list[str] = []
        warnings: list[str] = []

        if not offer_line:
            missing_reason = "no_usable_price"
            return OfferExtraction(
                thread_id=thread_id,
                extraction_method=ExtractionMethod.DETERMINISTIC,
                extraction_confidence=0.9,
                validation_status=ValidationStatus.INCOMPLETE,
                source_message=source_message,
                latest_message_text=latest_text,
                ambiguities=("no_price_bearing_sentence",),
                preliminary_missing_reasons=(missing_reason,),
                payment_terms=detect_terms(latest_text)["payment"],
                delivery_terms=detect_terms(latest_text)["delivery"],
                warranty_terms=detect_terms(latest_text)["warranty"],
                source_locations=("no price token found",),
            )

        offer_text = offer_line
        lower = offer_text.lower()
        is_correction = bool(re.search(r"(?i)\bcorrection\b|should read", offer_text))
        selection_rule: str = "single"
        tier_threshold = None
        selected_index: int | None = None
        superseded_amount = None
        superseded_currency = None
        candidate_scan_text = offer_text
        if is_correction:
            cut = re.search(r"(?i),\s*not\b", candidate_scan_text)
            if cut:
                superseded_matches = find_money(candidate_scan_text[cut.end() :])
                if superseded_matches:
                    superseded_amount = superseded_matches[0].amount
                    superseded_currency = superseded_matches[0].currency
                candidate_scan_text = candidate_scan_text[: cut.start()]
            selection_rule = "correction"
        candidates = tuple(find_money(candidate_scan_text))
        if not candidates:
            preliminary_missing.append("no_usable_price")

        if "between" in lower:
            selection_rule = "range_high"
        if "up to" in lower and "larger order" in lower:
            selection_rule = "tier_by_quantity"
            threshold_match = re.search(r"(?i)up\s+to\s+([\d,]+)", offer_text)
            tier_threshold = _parse_int(threshold_match.group(1)) if threshold_match else None
            if len(candidates) >= 2:
                if tier_threshold is None:
                    ambiguities.append("tier_threshold_missing")
                # The actual thread quantity selects the tier in normalisation.
            else:
                preliminary_missing.append("tier_prices_incomplete")
        elif len(candidates) > 1 and not is_correction and "between" not in lower:
            # Multiple prices without a recognised selection rule are unsafe.
            selection_rule = "none"
            ambiguities.append("multiple_prices_without_selection_rule")
            preliminary_missing.append("ambiguous_multiple_prices")
        elif len(candidates) == 1:
            selection_rule = "correction" if is_correction else "single"
            selected_index = 0

        if is_correction and candidates:
            selection_rule = "correction"
            selected_index = 0

        discount_percent = None
        discount_base = None
        list_match = re.search(
            r"(?i)(\d+(?:\.\d+)?)\s*%\s+discount\s+on\s+our\s+list\s+price\s+of",
            offer_text,
        )
        previous_match = re.search(
            r"(?i)further\s+(\d+(?:\.\d+)?)\s*%\s+off\s+our\s+previous\s+quote",
            offer_text,
        )
        if list_match:
            discount_percent = Decimal(list_match.group(1))
            discount_base = "list_price"
            # The list price is the amount following 'list price of'; if more than one
            # amount exists, do not silently pick one.
            list_money = find_money(offer_text[list_match.end() :])
            if len(list_money) == 1:
                candidates = (list_money[0],)
                selected_index = 0
                selection_rule = "single"
            else:
                ambiguities.append("list_price_amount_ambiguous")
                preliminary_missing.append("ambiguous_list_price")
        elif previous_match:
            discount_percent = Decimal(previous_match.group(1))
            discount_base = "previous_quote"
            candidates = ()
            selected_index = None
            selection_rule = "none"

        volume_percent = None
        volume_threshold = None
        volume_match = re.search(
            r"(?i)further\s+(\d+(?:\.\d+)?)\s*%\s+off\s+for\s+orders\s+of\s+([\d,]+)",
            offer_text,
        )
        if volume_match:
            volume_percent = Decimal(volume_match.group(1))
            volume_threshold = _parse_int(volume_match.group(2))

        free_units = None
        paid_units = None
        free_match = re.search(
            r"(?i)(\d+)\s+free\s+[a-z ]+?\s+for\s+every\s+(\d+)\s+you\s+pay\s+for",
            offer_text,
        )
        if free_match:
            free_units = _parse_int(free_match.group(1))
            paid_units = _parse_int(free_match.group(2))

        is_total = bool(re.search(r"(?i)\bin total\b|\btotal\s+for\s+all\b", offer_text))
        total_quantity = _parse_total_quantity(offer_text) if is_total else None
        if is_total and total_quantity is None:
            ambiguities.append("total_quantity_missing")
            preliminary_missing.append("total_quantity_missing")

        factor, unit_label, factor_warning = detect_unit_factor(offer_text)
        if unit_label and unit_label != "ream" and factor == 1:
            base_label = unit_label.split()[0] if unit_label.startswith("pack of") else unit_label
            resolved = self.category_factor_resolver(base_label)
            if resolved is not None:
                factor = resolved
        if unit_label == "ream":
            resolved = self.category_factor_resolver("ream")
            if resolved is None:
                preliminary_missing.append("ream_conversion_unknown")
                warnings.append("ream_conversion_unknown")
            else:
                factor = resolved
        if factor_warning and factor_warning != "ream_factor_requires_category":
            warnings.append(factor_warning)

        billing_period = detect_billing_period(offer_text)
        gst_treatment = detect_gst(offer_text)
        if gst_treatment == "ambiguous":
            ambiguities.append("gst_inclusion_ambiguous")
            preliminary_missing.append("gst_inclusion_ambiguous")
        if snapshot.normalized_period is not None and billing_period is None:
            preliminary_missing.append("subscription_billing_period_missing")
            ambiguities.append("subscription_price_without_billing_period")

        previous_quote_text = None
        previous_candidates: tuple[PriceCandidate, ...] = ()
        previous_factor = Decimal(1)
        previous_period = None
        previous_gst = "unspecified"
        if discount_base == "previous_quote":
            previous_quote_text, previous = parse_previous_quote(
                source_message, snapshot, self.category_factor_resolver
            )
            if previous is None or not previous.candidates:
                preliminary_missing.append("previous_supplier_quote_missing")
                ambiguities.append("previous_quote_reference_not_resolvable")
            else:
                previous_candidates = previous.candidates
                previous_factor = previous.unit_factor
                previous_period = previous.billing_period
                previous_gst = previous.gst_treatment
                # The latest sentence contains no standalone price; the prior supplier
                # quote is the price basis and has now been resolved. Its billing period
                # and currency also carry over to the revised offer.
                preliminary_missing = [
                    reason for reason in preliminary_missing if reason != "no_usable_price"
                ]
                if previous_period is not None:
                    preliminary_missing = [
                        reason
                        for reason in preliminary_missing
                        if reason != "subscription_billing_period_missing"
                    ]
                    ambiguities = [
                        item
                        for item in ambiguities
                        if item != "subscription_price_without_billing_period"
                    ]

        terms = detect_terms(latest_text)
        validation_status = ValidationStatus.VALID
        if preliminary_missing:
            if any(reason.startswith("ambiguous") for reason in preliminary_missing):
                validation_status = ValidationStatus.AMBIGUOUS
            else:
                validation_status = ValidationStatus.INCOMPLETE
        confidence = 0.99
        if ambiguities:
            confidence = 0.75
        if preliminary_missing:
            confidence = min(confidence, 0.55)
        if not candidates and discount_base != "previous_quote":
            confidence = min(confidence, 0.35)

        # Locate the offer line in the original source for provenance.
        search_start = source_message.rfind(latest_text)
        search_start = max(search_start, 0)
        span_start = source_message.find(offer_line, search_start)
        span_end = span_start + len(offer_line) if span_start >= 0 else None

        return OfferExtraction(
            thread_id=thread_id,
            extraction_method=ExtractionMethod.DETERMINISTIC,
            extraction_confidence=round(confidence, 4),
            validation_status=validation_status,
            source_message=source_message,
            latest_message_text=latest_text,
            offer_text=offer_text,
            offer_span_start=span_start if span_start >= 0 else None,
            offer_span_end=span_end,
            price_candidates=candidates,
            selection_rule=selection_rule,  # type: ignore[arg-type]
            selected_candidate_index=selected_index,
            tier_threshold=tier_threshold,
            is_correction=is_correction,
            superseded_amount=superseded_amount,
            superseded_currency=superseded_currency,
            is_total=is_total,
            total_quantity=total_quantity,
            quoted_unit_factor=factor,
            quoted_unit_label=unit_label,
            billing_period=billing_period,  # type: ignore[arg-type]
            gst_treatment=gst_treatment,  # type: ignore[arg-type]
            discount_percent=discount_percent,
            discount_base=discount_base,  # type: ignore[arg-type]
            volume_discount_percent=volume_percent,
            volume_threshold=volume_threshold,
            volume_applies=None,
            free_units=free_units,
            paid_units=paid_units,
            previous_quote_text=previous_quote_text,
            previous_quote_candidates=previous_candidates,
            previous_quote_unit_factor=previous_factor,
            previous_quote_billing_period=previous_period,  # type: ignore[arg-type]
            previous_quote_gst_treatment=previous_gst,  # type: ignore[arg-type]
            payment_terms=terms["payment"],
            delivery_terms=terms["delivery"],
            warranty_terms=terms["warranty"],
            ambiguities=tuple(ambiguities),
            preliminary_missing_reasons=tuple(dict.fromkeys(preliminary_missing)),
            warnings=tuple(warnings),
            source_locations=(f"latest supplier message, offer sentence: {offer_text}",),
        )
