"""Lightweight OpenTelemetry-compatible JSON tracing."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings


class Tracer:
    def __init__(self, run_id: str, path: str | Path | None = None) -> None:
        settings = get_settings()
        if path is None:
            trace_dir = PROJECT_ROOT / settings.observability.trace_directory
            trace_dir.mkdir(parents=True, exist_ok=True)
            path = trace_dir / f"{run_id}.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.trace_id = uuid.uuid4().hex
        self._span_stack: list[str] = []

    @contextmanager
    def span(self, name: str, attributes: dict | None = None) -> Iterator[dict]:
        span_id = uuid.uuid4().hex[:16]
        parent_id = self._span_stack[-1] if self._span_stack else None
        started = time.perf_counter()
        record = {
            "trace_id": self.trace_id,
            "span_id": span_id,
            "parent_span_id": parent_id,
            "run_id": self.run_id,
            "name": name,
            "start_time": datetime.now(UTC).isoformat(),
            "attributes": attributes or {},
        }
        self._span_stack.append(span_id)
        try:
            yield record
        finally:
            self._span_stack.pop()
            record["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            record["end_time"] = datetime.now(UTC).isoformat()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
