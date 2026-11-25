"""Extraction orchestration: deterministic default with optional grounded LLM assist."""
from __future__ import annotations

import os
from collections.abc import Callable
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from src.errors import ExtractionError, LLMProviderError
from src.extraction.deterministic_extractor import DeterministicExtractor
from src.extraction.extraction_schema import (
    ExtractionMethod,
    OfferExtraction,
    ValidationStatus,
)
from src.extraction.llm_extractor import LLMExtractor
from src.extraction.validation import validate_extraction_grounding
from src.knowledge.policy_snapshot import PolicySnapshot
from src.llm.provider import LLMConfig, LLMProvider, get_provider


class ExtractionOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    extraction: OfferExtraction
    requested_mode: str
    used_method: ExtractionMethod
    fallback_used: bool = False
    llm_calls: int = 0
    warnings: tuple[str, ...] = ()


class ExtractionOrchestrator:
    def __init__(
        self,
        *,
        category_factor_resolver: Callable[[str], Decimal | None],
        mode: str | None = None,
        provider: LLMProvider | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.mode = (mode or os.getenv("EXTRACTION_MODE") or "deterministic").strip().lower()
        self.config = llm_config or LLMConfig.from_env()
        self.provider = provider or get_provider(self.config)
        self.deterministic = DeterministicExtractor(category_factor_resolver=category_factor_resolver)
        self.category_factor_resolver = category_factor_resolver

    def extract(
        self,
        *,
        thread_id: str,
        source_message: str,
        snapshot: PolicySnapshot,
    ) -> ExtractionOutcome:
        deterministic = self.deterministic.extract(
            thread_id=thread_id, source_message=source_message, snapshot=snapshot
        )
        if self.mode in {"deterministic", "det", "off"}:
            return ExtractionOutcome(
                extraction=deterministic,
                requested_mode=self.mode,
                used_method=ExtractionMethod.DETERMINISTIC,
                fallback_used=False,
                warnings=deterministic.warnings,
            )
        if self.mode not in {"auto", "llm", "hybrid"}:
            return ExtractionOutcome(
                extraction=deterministic,
                requested_mode=self.mode,
                used_method=ExtractionMethod.DETERMINISTIC,
                fallback_used=True,
                warnings=("unknown extraction mode; deterministic fallback",),
            )
        # 'auto' trusts a complete deterministic parse and only asks the LLM for gaps.
        if self.mode == "auto" and deterministic.validation_status == ValidationStatus.VALID and deterministic.extraction_confidence >= 0.85:
            return ExtractionOutcome(
                extraction=deterministic,
                requested_mode=self.mode,
                used_method=ExtractionMethod.DETERMINISTIC,
                fallback_used=False,
                warnings=deterministic.warnings,
            )
        if not self.config.configured:
            return ExtractionOutcome(
                extraction=deterministic,
                requested_mode=self.mode,
                used_method=ExtractionMethod.DETERMINISTIC,
                fallback_used=True,
                warnings=("LLM requested but no provider configured; deterministic fallback",),
            )
        llm_extractor = LLMExtractor(
            provider=self.provider,
            config=self.config,
            category_factor_resolver=self.category_factor_resolver,
        )
        try:
            llm = llm_extractor.extract(
                thread_id=thread_id, source_message=source_message, snapshot=snapshot
            )
            validation = validate_extraction_grounding(llm, snapshot)
            if not validation.valid:
                raise ExtractionError("; ".join(validation.errors))
            if deterministic.validation_status == ValidationStatus.VALID:
                # A complete deterministic parse is the safer source of truth. Only
                # accept LLM output when it agrees on the material price/unit fields.
                if (
                    llm.selected_candidate.amount if llm.selected_candidate else None
                ) != (
                    deterministic.selected_candidate.amount if deterministic.selected_candidate else None
                ):
                    return ExtractionOutcome(
                        extraction=deterministic,
                        requested_mode=self.mode,
                        used_method=ExtractionMethod.DETERMINISTIC,
                        fallback_used=True,
                        llm_calls=llm_extractor.last_call_count,
                        warnings=("LLM disagreed with a complete deterministic parse; deterministic kept",),
                    )
            return ExtractionOutcome(
                extraction=llm,
                requested_mode=self.mode,
                used_method=ExtractionMethod.LLM,
                fallback_used=False,
                llm_calls=llm_extractor.last_call_count,
                warnings=tuple(validation.warnings),
            )
        except (LLMProviderError, ExtractionError, ValueError) as exc:
            return ExtractionOutcome(
                extraction=deterministic,
                requested_mode=self.mode,
                used_method=ExtractionMethod.DETERMINISTIC,
                fallback_used=True,
                llm_calls=getattr(llm_extractor, "last_call_count", 0),
                warnings=(f"LLM extraction failed or was rejected; deterministic fallback: {exc}",),
            )
