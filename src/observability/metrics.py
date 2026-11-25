"""Measured run metrics; no synthetic values are inserted."""
from __future__ import annotations

import statistics
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ThreadMetrics(BaseModel):
    model_config = ConfigDict(frozen=False)

    thread_id: str
    label: str | None = None
    applied_rule: str | None = None
    verification_status: str | None = None
    extraction_method: str | None = None
    extraction_confidence: float | None = None
    retrieval_confidence: float | None = None
    evidence_count: int = 0
    llm_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    stage_trace: tuple[str, ...] = ()


class RunMetrics(BaseModel):
    model_config = ConfigDict(frozen=False)

    run_id: str
    started_at: str
    finished_at: str | None = None
    total_threads: int = 0
    label_counts: dict[str, int] = Field(default_factory=dict)
    rule_counts: dict[str, int] = Field(default_factory=dict)
    verification_counts: dict[str, int] = Field(default_factory=dict)
    extraction_method_counts: dict[str, int] = Field(default_factory=dict)
    avg_retrieval_confidence: float | None = None
    avg_evidence_count: float | None = None
    total_llm_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None
    total_time_ms: float | None = None
    avg_time_ms: float | None = None
    stage_avg_ms: dict[str, float] = Field(default_factory=dict)
    warnings_count: int = 0
    failures_count: int = 0
    threads_per_second: float | None = None
    threads: list[ThreadMetrics] = Field(default_factory=list)


class MetricsCollector:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.started_at = datetime.now(UTC)
        self.threads: list[ThreadMetrics] = []

    def add(self, metrics: ThreadMetrics) -> None:
        self.threads.append(metrics)

    def summary(self) -> RunMetrics:
        finished = datetime.now(UTC)
        label_counts = Counter(t.label or "UNKNOWN" for t in self.threads)
        rule_counts = Counter(t.applied_rule or "UNKNOWN" for t in self.threads)
        verification_counts = Counter(t.verification_status or "UNKNOWN" for t in self.threads)
        method_counts = Counter(t.extraction_method or "UNKNOWN" for t in self.threads)
        retrieval_confidences = [t.retrieval_confidence for t in self.threads if t.retrieval_confidence is not None]
        evidence_counts = [t.evidence_count for t in self.threads]
        stage_values: dict[str, list[float]] = {}
        for thread in self.threads:
            for stage, duration in thread.timings_ms.items():
                stage_values.setdefault(stage, []).append(duration)
        total_llm = sum(t.llm_calls for t in self.threads)
        prompt_tokens = (
            sum(t.prompt_tokens or 0 for t in self.threads)
            if any(t.prompt_tokens is not None for t in self.threads)
            else None
        )
        completion_tokens = (
            sum(t.completion_tokens or 0 for t in self.threads)
            if any(t.completion_tokens is not None for t in self.threads)
            else None
        )
        costs = [t.estimated_cost_usd for t in self.threads if t.estimated_cost_usd is not None]
        total_time = (finished - self.started_at).total_seconds() * 1000.0
        return RunMetrics(
            run_id=self.run_id,
            started_at=self.started_at.isoformat(),
            finished_at=finished.isoformat(),
            total_threads=len(self.threads),
            label_counts=dict(label_counts),
            rule_counts=dict(rule_counts),
            verification_counts=dict(verification_counts),
            extraction_method_counts=dict(method_counts),
            avg_retrieval_confidence=(
                round(statistics.mean(retrieval_confidences), 6) if retrieval_confidences else None
            ),
            avg_evidence_count=(round(statistics.mean(evidence_counts), 3) if evidence_counts else None),
            total_llm_calls=total_llm,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=(round(sum(costs), 6) if costs else None),
            total_time_ms=round(total_time, 3),
            avg_time_ms=(round(total_time / len(self.threads), 3) if self.threads else None),
            stage_avg_ms={
                stage: round(statistics.mean(values), 3) for stage, values in sorted(stage_values.items())
            },
            warnings_count=sum(len(t.warnings) for t in self.threads),
            failures_count=sum(len(t.failures) for t in self.threads),
            threads_per_second=(
                round(len(self.threads) / (total_time / 1000.0), 3) if total_time > 0 else None
            ),
            threads=self.threads,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.summary().model_dump(mode="json")
