"""Audit persistence: one JSONL record and one SQLite row per thread."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuditRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    thread_id: str
    created_at: str
    input: dict[str, Any]
    final_label: str
    safe_failure: bool = False
    decision: dict[str, Any] | None = None
    extraction: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None
    policy_snapshot: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    security: dict[str, Any] | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    stage_trace: tuple[str, ...] = ()


class AuditStore:
    def __init__(self, directory: str | Path, run_id: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.jsonl_path = self.directory / "audit.jsonl"
        self.sqlite_path = self.directory / "audit.sqlite"
        self._seen: set[str] = set()
        self._connection = sqlite3.connect(self.sqlite_path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_records (
                thread_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                label TEXT NOT NULL,
                action TEXT,
                reason TEXT,
                verification_status TEXT,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_snapshots (
                thread_id TEXT PRIMARY KEY,
                snapshot_id TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                record_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def write(self, record: AuditRecord) -> None:
        if record.thread_id in self._seen:
            raise ValueError(f"duplicate audit record for {record.thread_id}")
        payload = record.model_dump(mode="json")
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        decision = record.decision or {}
        verification = record.verification or {}
        self._connection.execute(
            """
            INSERT OR REPLACE INTO audit_records
            (thread_id, run_id, label, action, reason, verification_status, record_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.thread_id,
                record.run_id,
                record.final_label,
                decision.get("action"),
                decision.get("reason"),
                verification.get("status"),
                json.dumps(payload, ensure_ascii=False, default=str),
                record.created_at,
            ),
        )
        if record.policy_snapshot:
            snapshot = record.policy_snapshot
            self._connection.execute(
                """
                INSERT OR REPLACE INTO policy_snapshots
                (thread_id, snapshot_id, policy_version, record_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.thread_id,
                    snapshot.get("snapshot_id", ""),
                    snapshot.get("policy_version", ""),
                    json.dumps(snapshot, ensure_ascii=False, default=str),
                ),
            )
        self._connection.commit()
        self._seen.add(record.thread_id)

    def has(self, thread_id: str) -> bool:
        return thread_id in self._seen

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> AuditStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
