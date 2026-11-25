"""Optional LLM-based structured extraction.

LLM output is converted into the same OfferExtraction schema as the deterministic
parser, grounded against the source message, and then passed through the same
normalizer, decision engine and verifier. It never decides the action.
"""
from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from pydantic import ValidationError

from src.errors import ExtractionError
from src.extraction.deterministic_extractor import (
    detect_billing_period,
    detect_gst,
    detect_terms,
    detect_unit_factor,
    find_money,
    normalise_currency,
    parse_previous_quote,
)
from src.extraction.extraction_schema import (
    ExtractionMethod,
    LLMExtractionResponse,
    OfferExtraction,
    PriceCandidate,
    ValidationStatus,
)
from src.knowledge.policy_snapshot import PolicySnapshot
from src.llm.cache import LLMCache
from src.llm.prompts import SYSTEM_PROMPT, build_extraction_prompt
from src.llm.provider import LLMConfig, LLMProvider, get_provider
from src.utils.text import latest_supplier_body


class LLMExtractor:
    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        config: LLMConfig | None = None,
        cache: LLMCache | None = None,
        category_factor_resolver: Callable[[str], Decimal | None] | None = None,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self.provider = provider or get_provider(self.config)
        self.cache = cache if cache is not None else LLMCache()
        self.category_factor_resolver = category_factor_resolver or (lambda _label: None)
        self.last_call_count = 0

    def extract(
        self,
        *,
        thread_id: str,
        source_message: str,
        snapshot: PolicySnapshot,
    ) -> OfferExtraction:
        latest = latest_supplier_body(source_message)
        prompt = build_extraction_prompt(thread_context=source_message, latest_supplier_message=latest)
        cached = self.cache.get(
            model=self.config.model or self.provider.name,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        self.last_call_count = 0
        if cached is not None:
            raw_response = cached
        else:
            self.last_call_count = 1
            raw_response = self.provider.complete_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                schema=LLMExtractionResponse.model_json_schema(),
            )
            self.cache.put(
                model=self.config.model or self.provider.name,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                value=raw_response,
            )
        try:
            response = LLMExtractionResponse.model_validate(raw_response)
        except ValidationError as exc:
            raise ExtractionError(f"LLM extraction failed schema validation: {exc}") from exc
        return self._to_extraction(
            thread_id=thread_id,
            source_message=source_message,
            latest=latest,
            response=response,
        )

    def _to_extraction(
        self,
        *,
        thread_id: str,
        source_message: str,
        latest: str,
        response: LLMExtractionResponse,
    ) -> OfferExtraction:
        source_text = response.source_text
        if source_text not in source_message:
            raise ExtractionError("LLM source_text is not present verbatim in the thread")
        if not response.usable_price or response.offered_price is None:
            if response.discount_base != "previous_quote":
                return OfferExtraction(
                    thread_id=thread_id,
                    extraction_method=ExtractionMethod.LLM,
                    extraction_confidence=response.confidence,
                    validation_status=ValidationStatus.INCOMPLETE,
                    source_message=source_message,
                    latest_message_text=latest,
                    offer_text=source_text,
                    preliminary_missing_reasons=("no_usable_price",),
                    ambiguities=response.ambiguities,
                    payment_terms=detect_terms(latest)["payment"],
                    delivery_terms=detect_terms(latest)["delivery"],
                    warranty_terms=detect_terms(latest)["warranty"],
                    source_locations=("LLM extraction grounded in source text",),
                )
        parsed_matches = find_money(source_text)
        candidate_amounts: list[Decimal] = []
        if response.tier_prices:
            candidate_amounts = list(response.tier_prices)
        elif response.discount_base == "list_price" and response.list_price is not None:
            candidate_amounts = [response.list_price]
        elif response.offered_price is not None:
            candidate_amounts = [response.offered_price]
        grounded: list[PriceCandidate] = []
        used_spans: set[tuple[int, int]] = set()
        for amount in candidate_amounts:
            match = next(
                (
                    candidate
                    for candidate in parsed_matches
                    if candidate.amount == amount and (candidate.start, candidate.end) not in used_spans
                ),
                None,
            )
            if match is None:
                raise ExtractionError(f"LLM price {amount} is not present in its cited source text")
            response_currency = normalise_currency(response.currency)
            if (
                response_currency is not None
                and match.currency is not None
                and response_currency != match.currency
            ):
                raise ExtractionError(
                    "LLM currency does not match the currency token in its cited source text"
                )
            grounded.append(
                PriceCandidate(
                    amount=match.amount,
                    currency=response_currency or match.currency,
                    raw_text=match.raw_text,
                    start=match.start,
                    end=match.end,
                    numeric_form=match.numeric_form,
                )
            )
            used_spans.add((match.start, match.end))
        candidates = tuple(grounded)
        factor, label, _ = detect_unit_factor(source_text)
        if label and factor == 1:
            base_label = label.split()[0] if label.startswith("pack of") else label
            factor = self.category_factor_resolver(base_label) or factor
        if label == "ream":
            factor = self.category_factor_resolver("ream") or factor
        period = response.billing_period or detect_billing_period(source_text)
        if response.gst_included is True:
            gst = "included"
        elif response.gst_included is False:
            gst = "excluded"
        else:
            gst = detect_gst(source_text)
        is_total = response.is_total
        total_quantity = response.total_quantity
        discount_base = response.discount_base
        discount_percent = response.discount_percent
        if discount_base == "previous_quote" and discount_percent is None:
            raise ExtractionError("LLM previous-quote discount lacks a percentage")
        previous_text = None
        previous_candidates: tuple[PriceCandidate, ...] = ()
        previous_factor = Decimal(1)
        previous_period = None
        previous_gst = "unspecified"
        if discount_base == "previous_quote":
            previous_text, previous = parse_previous_quote(
                source_message, None, self.category_factor_resolver
            )
            if previous is None:
                raise ExtractionError("LLM previous-quote reference cannot be resolved to supplier text")
            previous_candidates = previous.candidates
            previous_factor = previous.unit_factor
            previous_period = previous.billing_period
            previous_gst = previous.gst_treatment
        selection_rule = "single"
        selected_index = 0 if candidates else None
        if response.tier_prices and len(response.tier_prices) >= 2:
            selection_rule = "tier_by_quantity"
            selected_index = None
        elif discount_base == "previous_quote":
            selection_rule = "none"
            selected_index = None
        if response.is_correction:
            selection_rule = "correction"
            selected_index = 0 if candidates else None
        ambiguities = list(response.ambiguities)
        preliminary_missing: list[str] = []
        if not candidates and discount_base != "previous_quote":
            preliminary_missing.append("no_usable_price")
        validation_status = ValidationStatus.VALID
        if preliminary_missing:
            validation_status = ValidationStatus.INCOMPLETE
        if ambiguities:
            validation_status = ValidationStatus.AMBIGUOUS
        terms = detect_terms(latest)
        return OfferExtraction(
            thread_id=thread_id,
            extraction_method=ExtractionMethod.LLM,
            extraction_confidence=response.confidence,
            validation_status=validation_status,
            source_message=source_message,
            latest_message_text=latest,
            offer_text=source_text,
            price_candidates=candidates,
            selection_rule=selection_rule,  # type: ignore[arg-type]
            selected_candidate_index=selected_index,
            tier_threshold=response.tier_threshold,
            is_correction=response.is_correction,
            superseded_amount=response.superseded_price,
            is_total=is_total,
            total_quantity=total_quantity,
            quoted_unit_factor=factor,
            quoted_unit_label=label,
            billing_period=period,  # type: ignore[arg-type]
            gst_treatment=gst,  # type: ignore[arg-type]
            discount_percent=discount_percent,
            discount_base=discount_base,  # type: ignore[arg-type]
            volume_discount_percent=response.volume_discount_percent,
            volume_threshold=response.volume_threshold,
            free_units=response.free_units,
            paid_units=response.paid_units,
            previous_quote_text=previous_text,
            previous_quote_candidates=previous_candidates,
            previous_quote_unit_factor=previous_factor,
            previous_quote_billing_period=previous_period,  # type: ignore[arg-type]
            previous_quote_gst_treatment=previous_gst,  # type: ignore[arg-type]
            payment_terms=terms["payment"],
            delivery_terms=terms["delivery"],
            warranty_terms=terms["warranty"],
            ambiguities=tuple(ambiguities),
            preliminary_missing_reasons=tuple(preliminary_missing),
            source_locations=("LLM extraction grounded in cited source_text",),
        )
