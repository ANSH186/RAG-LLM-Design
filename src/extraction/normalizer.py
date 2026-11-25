"""Policy-driven offer normalisation using Decimal arithmetic.

No value is rounded before the final price comparison. The normalizer does not make a
procurement decision; it only converts a structured supplier offer into INR, excluding
GST, per the category's normalized unit.
"""
from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from src.extraction.extraction_schema import (
    NormalizationStep,
    NormalizedOffer,
    OfferExtraction,
    PriceCandidate,
)
from src.knowledge.policy_snapshot import PolicySnapshot

FactorResolver = Callable[[str], Decimal | None]


class OfferNormalizer:
    def __init__(self, snapshot: PolicySnapshot, factor_resolver: FactorResolver | None = None) -> None:
        self.snapshot = snapshot
        self.factor_resolver = factor_resolver or (lambda _label: None)

    def normalize(self, extraction: OfferExtraction, *, quantity: int) -> NormalizedOffer:
        missing: list[str] = list(extraction.preliminary_missing_reasons)
        warnings: list[str] = list(extraction.warnings)
        steps: list[NormalizationStep] = []

        if extraction.discount_base == "previous_quote":
            return self._normalize_previous_quote(extraction, quantity, missing, warnings, steps)

        candidates = extraction.price_candidates
        if not candidates:
            if "no_usable_price" not in missing:
                missing.append("no_usable_price")
            return NormalizedOffer(
                is_usable=False,
                normalized_price_inr_excl_gst=None,
                quantity_used=quantity,
                gst_rate=self.snapshot.gst_rate,
                missing_information_reasons=tuple(dict.fromkeys(missing)),
                warnings=tuple(dict.fromkeys(warnings)),
                steps=tuple(steps),
                confidence=min(0.4, extraction.extraction_confidence),
            )

        rule = extraction.selection_rule
        selected: list[PriceCandidate] = []
        selection_detail: str | None = None
        if rule == "range_high":
            selected = list(candidates)
            selection_detail = "range: evaluate the higher end after normalisation"
        elif rule == "tier_by_quantity":
            if extraction.tier_threshold is None or len(candidates) < 2:
                missing.append("tier_structure_incomplete")
                return self._unusable(extraction, quantity, missing, warnings, steps)
            threshold = extraction.tier_threshold
            index = 0 if quantity <= threshold else 1
            selected = [candidates[index]]
            selection_detail = (
                f"tier selected at quantity {quantity} (threshold {threshold}, upper bound inclusive)"
            )
        elif rule == "correction":
            if extraction.selected_candidate_index is not None:
                selected = [candidates[extraction.selected_candidate_index]]
            else:
                selected = [candidates[0]]
            selection_detail = "correction replaces the price it corrects"
        elif rule == "single":
            if len(candidates) == 1:
                selected = [candidates[0]]
            elif extraction.selected_candidate_index is not None:
                selected = [candidates[extraction.selected_candidate_index]]
            else:
                missing.append("ambiguous_multiple_prices")
                return self._unusable(extraction, quantity, missing, warnings, steps)
            selection_detail = "single offer price"
        else:
            missing.append("ambiguous_multiple_prices")
            return self._unusable(extraction, quantity, missing, warnings, steps)

        normalised: list[tuple[Decimal, PriceCandidate, list[NormalizationStep]]] = []
        for candidate in selected:
            value, candidate_steps = self._normalise_candidate(
                candidate,
                extraction=extraction,
                quantity=quantity,
                missing=missing,
                warnings=warnings,
            )
            if value is not None:
                normalised.append((value, candidate, candidate_steps))
        if not normalised:
            return self._unusable(extraction, quantity, missing, warnings, steps)
        if rule == "range_high":
            value, chosen, candidate_steps = max(normalised, key=lambda item: item[0])
        else:
            value, chosen, candidate_steps = normalised[0]
        steps.extend(candidate_steps)
        if any(reason.startswith("unsupported_currency") for reason in missing):
            return self._unusable(extraction, quantity, missing, warnings, steps)
        if any(reason == "ambiguous_currency" for reason in missing):
            return self._unusable(extraction, quantity, missing, warnings, steps)
        if missing:
            return self._unusable(extraction, quantity, missing, warnings, steps, chosen.currency, value)

        confidence = extraction.extraction_confidence
        if warnings:
            confidence = min(confidence, 0.85)
        return NormalizedOffer(
            is_usable=True,
            normalized_price_inr_excl_gst=value,
            currency=chosen.currency,
            exchange_rate=self._rate(chosen.currency),
            amount_before_conversion=chosen.amount,
            quantity_used=extraction.total_quantity or quantity,
            gst_rate=self.snapshot.gst_rate,
            gst_included=extraction.gst_treatment == "included",
            billing_period_quoted=extraction.billing_period,
            category_billing_period=self.snapshot.normalized_period,
            selected_raw_price=chosen.amount,
            selection_detail=selection_detail,
            missing_information_reasons=(),
            warnings=tuple(dict.fromkeys(warnings)),
            steps=tuple(steps),
            confidence=round(confidence, 4),
        )

    def _normalise_candidate(
        self,
        candidate: PriceCandidate,
        *,
        extraction: OfferExtraction,
        quantity: int,
        missing: list[str],
        warnings: list[str],
    ) -> tuple[Decimal | None, list[NormalizationStep]]:
        steps: list[NormalizationStep] = []
        value = candidate.amount
        steps.append(
            NormalizationStep(
                rule="base_price",
                detail=f"quoted {candidate.currency or 'unknown currency'} amount",
                input_value=str(candidate.amount),
                output_value=str(value),
            )
        )

        # List-price discount: the amount captured is the list price.
        if extraction.discount_base == "list_price" and extraction.discount_percent is not None:
            percent = extraction.discount_percent
            if percent < 0 or percent > 100:
                missing.append("invalid_discount_percentage")
                return None, steps
            before = value
            value = value * (Decimal(1) - (percent / Decimal(100)))
            steps.append(
                NormalizationStep(
                    rule="list_price_discount",
                    detail=f"{percent}% off list price",
                    input_value=str(before),
                    output_value=str(value),
                )
            )

        # Volume discount only if the thread quantity meets the threshold (inclusive).
        if extraction.volume_discount_percent is not None and extraction.volume_threshold is not None:
            applies = quantity >= extraction.volume_threshold
            steps.append(
                NormalizationStep(
                    rule="volume_discount",
                    detail=(
                        f"threshold {extraction.volume_threshold}, quantity {quantity}: "
                        f"{'applied' if applies else 'not applied'}"
                    ),
                    input_value=str(value),
                    output_value=str(value),
                )
            )
            if applies:
                before = value
                value = value * (
                    Decimal(1) - extraction.volume_discount_percent / Decimal(100)
                )
                steps[-1] = steps[-1].model_copy(update={"output_value": str(value)})

        # Free units: effective price across paid+free units.
        if extraction.free_units is not None and extraction.paid_units is not None:
            if extraction.paid_units <= 0 or extraction.free_units < 0:
                missing.append("invalid_free_units")
                return None, steps
            before = value
            value = value * Decimal(extraction.paid_units) / Decimal(
                extraction.paid_units + extraction.free_units
            )
            steps.append(
                NormalizationStep(
                    rule="free_units",
                    detail=(
                        f"{extraction.free_units} free per {extraction.paid_units} paid; "
                        f"effective paid/paid+free ratio"
                    ),
                    input_value=str(before),
                    output_value=str(value),
                )
            )

        # Lots/packs/dozens/reams: quoted price covers quoted_unit_factor normalized units.
        factor = extraction.quoted_unit_factor
        if factor == 0:
            missing.append("invalid_unit_factor")
            return None, steps
        if factor != 1:
            before = value
            value = value / factor
            steps.append(
                NormalizationStep(
                    rule="subunit_conversion",
                    detail=f"quoted unit {extraction.quoted_unit_label!r} contains {factor} normalized units",
                    input_value=str(before),
                    output_value=str(value),
                )
            )

        # Totals are divided by the number of units they cover.
        if extraction.is_total:
            units = extraction.total_quantity or quantity
            if units <= 0:
                missing.append("total_quantity_missing")
                return None, steps
            before = value
            value = value / Decimal(units)
            steps.append(
                NormalizationStep(
                    rule="total_price",
                    detail=f"total divided by {units} units",
                    input_value=str(before),
                    output_value=str(value),
                )
            )

        # Billing period normalisation.
        period = extraction.billing_period
        value = self._convert_period(value, period, missing, steps)
        if value is None:
            return None, steps

        # Currency conversion.
        currency = candidate.currency
        if currency is None:
            missing.append("ambiguous_currency")
            return None, steps
        rate = self._rate(currency)
        if rate is None:
            missing.append(f"unsupported_currency:{currency}")
            return None, steps
        before = value
        value = value * rate
        steps.append(
            NormalizationStep(
                rule="currency_conversion",
                detail=f"{currency} at {rate} INR per unit",
                input_value=str(before),
                output_value=str(value),
            )
        )

        # GST inclusion: remove GST to compare on an ex-GST basis.
        gst = extraction.gst_treatment
        if gst == "included":
            before = value
            value = value / (Decimal(1) + self.snapshot.gst_rate)
            steps.append(
                NormalizationStep(
                    rule="gst_exclusion",
                    detail=f"divide by 1 + GST rate {self.snapshot.gst_rate}",
                    input_value=str(before),
                    output_value=str(value),
                )
            )
        elif gst == "excluded" or gst == "unspecified":
            steps.append(
                NormalizationStep(
                    rule="gst_exclusion",
                    detail=f"price treated as {gst} GST",
                    input_value=str(value),
                    output_value=str(value),
                )
            )
        else:
            missing.append("gst_inclusion_ambiguous")
            return None, steps
        return value, steps

    def _normalize_previous_quote(
        self,
        extraction: OfferExtraction,
        quantity: int,
        missing: list[str],
        warnings: list[str],
        steps: list[NormalizationStep],
    ) -> NormalizedOffer:
        previous_candidates = extraction.previous_quote_candidates
        if not previous_candidates:
            if "previous_supplier_quote_missing" not in missing:
                missing.append("previous_supplier_quote_missing")
            return self._unusable(extraction, quantity, missing, warnings, steps)
        previous = previous_candidates[0]
        # Previous quotes in the playbook are simple unit quotes. Normalise them with
        # their own terms before applying the current further discount.
        prev_value = previous.amount
        steps.append(
            NormalizationStep(
                rule="previous_quote_base",
                detail="previous supplier quote",
                input_value=str(prev_value),
                output_value=str(prev_value),
            )
        )
        if extraction.previous_quote_unit_factor != 1:
            before = prev_value
            prev_value = prev_value / extraction.previous_quote_unit_factor
            steps.append(
                NormalizationStep(
                    rule="previous_quote_unit",
                    detail=f"previous quote factor {extraction.previous_quote_unit_factor}",
                    input_value=str(before),
                    output_value=str(prev_value),
                )
            )
        prev_value = self._convert_period(
            prev_value, extraction.previous_quote_billing_period, missing, steps
        )
        if prev_value is None:
            return self._unusable(extraction, quantity, missing, warnings, steps)
        rate = self._rate(previous.currency)
        if rate is None:
            missing.append(f"unsupported_currency:{previous.currency}")
            return self._unusable(extraction, quantity, missing, warnings, steps)
        before = prev_value
        prev_value = prev_value * rate
        steps.append(
            NormalizationStep(
                rule="previous_quote_currency",
                detail=f"{previous.currency} at {rate} INR per unit",
                input_value=str(before),
                output_value=str(prev_value),
            )
        )
        if extraction.previous_quote_gst_treatment == "included":
            before = prev_value
            prev_value = prev_value / (Decimal(1) + self.snapshot.gst_rate)
            steps.append(
                NormalizationStep(
                    rule="previous_quote_gst",
                    detail=f"previous quote GST-excluded at {self.snapshot.gst_rate}",
                    input_value=str(before),
                    output_value=str(prev_value),
                )
            )
        elif extraction.previous_quote_gst_treatment == "ambiguous":
            missing.append("previous_quote_gst_ambiguous")
            return self._unusable(extraction, quantity, missing, warnings, steps)
        if extraction.discount_percent is None or extraction.discount_percent < 0:
            missing.append("previous_quote_discount_missing")
            return self._unusable(extraction, quantity, missing, warnings, steps)
        before = prev_value
        prev_value = prev_value * (Decimal(1) - extraction.discount_percent / Decimal(100))
        steps.append(
            NormalizationStep(
                rule="previous_quote_discount",
                detail=f"further {extraction.discount_percent}% off supplier's own previous quote",
                input_value=str(before),
                output_value=str(prev_value),
            )
        )
        if missing:
            return self._unusable(
                extraction, quantity, missing, warnings, steps, previous.currency, prev_value
            )
        return NormalizedOffer(
            is_usable=True,
            normalized_price_inr_excl_gst=prev_value,
            currency=previous.currency,
            exchange_rate=self._rate(previous.currency),
            amount_before_conversion=previous.amount,
            quantity_used=quantity,
            gst_rate=self.snapshot.gst_rate,
            gst_included=extraction.previous_quote_gst_treatment == "included",
            billing_period_quoted=extraction.previous_quote_billing_period,
            category_billing_period=self.snapshot.normalized_period,
            selected_raw_price=previous.amount,
            selection_detail="further discount off supplier's own previous quote",
            missing_information_reasons=(),
            warnings=tuple(dict.fromkeys(warnings)),
            steps=tuple(steps),
            confidence=round(min(extraction.extraction_confidence, 0.9), 4),
        )

    def _convert_period(
        self,
        value: Decimal,
        period: str | None,
        missing: list[str],
        steps: list[NormalizationStep],
    ) -> Decimal | None:
        target = self.snapshot.normalized_period
        if target is None and period is None:
            return value
        if target is None and period is not None:
            missing.append("unexpected_billing_period")
            return None
        if target is not None and period is None:
            if "subscription_billing_period_missing" not in missing:
                missing.append("subscription_billing_period_missing")
            return None
        if period == target:
            return value
        before = value
        if period == "month" and target == "year":
            value = value * Decimal(12)
            detail = "monthly to yearly x12"
        elif period == "year" and target == "month":
            value = value / Decimal(12)
            detail = "yearly to monthly /12"
        else:
            missing.append("unsupported_billing_period_conversion")
            return None
        steps.append(
            NormalizationStep(
                rule="billing_period",
                detail=detail,
                input_value=str(before),
                output_value=str(value),
            )
        )
        return value

    def _rate(self, currency: str | None) -> Decimal | None:
        if currency is None:
            return None
        try:
            return self.snapshot.effective_rate(currency)
        except KeyError:
            return None

    def _unusable(
        self,
        extraction: OfferExtraction,
        quantity: int,
        missing: list[str],
        warnings: list[str],
        steps: list[NormalizationStep],
        currency: str | None = None,
        value: Decimal | None = None,
    ) -> NormalizedOffer:
        return NormalizedOffer(
            is_usable=False,
            normalized_price_inr_excl_gst=None,
            currency=currency,
            exchange_rate=self._rate(currency),
            quantity_used=quantity,
            gst_rate=self.snapshot.gst_rate,
            missing_information_reasons=tuple(dict.fromkeys(missing)),
            warnings=tuple(dict.fromkeys(warnings)),
            steps=tuple(steps),
            confidence=min(0.5, extraction.extraction_confidence),
        )
