"""End-to-end procurement pipeline built on the LangGraph workflow."""
from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.audit import AuditRecord, AuditStore
from src.config import (
    EnvironmentConfig,
    ensure_runtime_dirs,
    get_environment,
    get_settings,
)
from src.evidence.context_builder import ContextBuilder
from src.evidence.source_registry import SourceRegistry
from src.knowledge.ingestion import load_knowledge
from src.llm.provider import LLMConfig, get_provider
from src.observability.logging import get_logger
from src.observability.metrics import MetricsCollector, ThreadMetrics
from src.observability.tracing import Tracer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.workflow.graph import build_graph
from src.workflow.state import PipelineServices


class ProcurementPipeline:
    def __init__(
        self,
        *,
        environment_code: str | None = None,
        run_id: str | None = None,
        extraction_mode: str | None = None,
        output_root: str | Path | None = None,
        audit_root: str | Path | None = None,
        trace_path: str | Path | None = None,
        llm_config: LLMConfig | None = None,
        provider=None,
    ) -> None:
        self.env: EnvironmentConfig = get_environment(environment_code)
        ensure_runtime_dirs(self.env)
        self.settings = get_settings()
        self.logger = get_logger("procurement.pipeline")
        self.run_id = run_id or os.getenv("RUN_ID") or (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        self.output_root = Path(output_root) if output_root else self.env.resolved(self.env.output_dir)
        self.run_output_dir = self.output_root / "runs" / self.run_id
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge = load_knowledge(self.env)
        self.registry = SourceRegistry(self.knowledge)
        self.retriever = HybridRetriever(self.knowledge, settings=self.settings)
        self.context_builder = ContextBuilder(self.registry)
        self.llm_config = llm_config or LLMConfig.from_env()
        self.llm_provider = provider or get_provider(self.llm_config)
        self.extraction_mode = (extraction_mode or os.getenv("EXTRACTION_MODE") or "deterministic").lower()
        self.metrics = MetricsCollector(self.run_id)
        self.tracer = Tracer(self.run_id, path=trace_path)
        audit_base = (
            Path(audit_root) if audit_root is not None else self.env.resolved(self.env.audit_dir)
        )
        audit_dir = audit_base / self.run_id
        self.audit_store = AuditStore(audit_dir, self.run_id)
        self.services = PipelineServices(
            run_id=self.run_id,
            knowledge=self.knowledge,
            registry=self.registry,
            retriever=self.retriever,
            context_builder=self.context_builder,
            extraction_mode=self.extraction_mode,
            llm_provider=self.llm_provider,
            llm_config=self.llm_config,
            metrics=self.metrics,
            tracer=self.tracer,
            audit_store=self.audit_store,
        )
        self.graph = build_graph(self.services)

    def process(self, row: dict[str, Any]) -> dict[str, Any]:
        thread_id = str(row.get("thread_id", "unknown"))
        state: dict[str, Any] = {
            "run_id": self.run_id,
            "thread_row": dict(row),
            "thread_started_perf": time.perf_counter(),
            "attempts": 1,
            "warnings": (),
            "errors": (),
            "timings_ms": {},
            "stage_trace": (),
        }
        try:
            result = self.graph.invoke(
                state,
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 40},
            )
            if not isinstance(result, dict):
                raise RuntimeError(f"workflow returned unexpected type {type(result).__name__}")
            if not result.get("final_label"):
                raise RuntimeError("workflow completed without a final label")
            return result
        except Exception as exc:
            self.logger.exception(
                "thread pipeline failure",
                extra={"thread_id": thread_id, "run_id": self.run_id, "event": "thread_failure"},
            )
            return self._fallback_failure(row, exc, started=state["thread_started_perf"])

    def _fallback_failure(
        self, row: dict[str, Any], exc: Exception, *, started: float | None = None
    ) -> dict[str, Any]:
        thread_id = str(row.get("thread_id", "unknown"))
        label = "ESCALATE:MISSING_INFO:"
        error = f"pipeline_exception:{type(exc).__name__}:{exc}"
        elapsed = round((time.perf_counter() - started) * 1000.0, 3) if started else 0.0
        timings = {"failover": elapsed}
        if not self.audit_store.has(thread_id):
            record = AuditRecord(
                run_id=self.run_id,
                thread_id=thread_id,
                created_at=datetime.now(UTC).isoformat(),
                input=dict(row),
                final_label=label,
                safe_failure=True,
                errors=(error,),
                timings_ms=timings,
                stage_trace=("failover",),
            )
            self.audit_store.write(record)
            self.metrics.add(
                ThreadMetrics(
                    thread_id=thread_id,
                    label=label,
                    verification_status="FAILED",
                    failures=(error,),
                    timings_ms=timings,
                    stage_trace=("failover",),
                )
            )
        return {
            "thread_id": thread_id,
            "final_label": label,
            "safe_failure": True,
            "errors": (error,),
            "timings_ms": timings,
            "stage_trace": ("failover",),
        }

    def run_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        total = len(frame)
        for index, (_, row) in enumerate(frame.iterrows(), start=1):
            results.append(self.process(row.to_dict()))
            if progress:
                progress(index, total)
        return results

    def labels_from_results(self, results: list[dict[str, Any]]) -> dict[str, str]:
        labels: dict[str, str] = {}
        for result in results:
            thread_id = str(result.get("thread_id") or (result.get("thread_row") or {}).get("thread_id"))
            label = result.get("final_label")
            if not thread_id or not label:
                raise RuntimeError(f"missing final output for thread {thread_id!r}")
            if thread_id in labels:
                raise RuntimeError(f"duplicate output for thread {thread_id}")
            labels[thread_id] = label
        return labels

    def close(self) -> None:
        self.audit_store.close()

    def __enter__(self) -> ProcurementPipeline:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
