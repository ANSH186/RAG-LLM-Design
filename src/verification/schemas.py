"""Verification result models."""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class VerificationStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    PASSED_AFTER_CORRECTION = "PASSED_AFTER_CORRECTION"
    SAFE_ESCALATED = "SAFE_ESCALATED"


class VerificationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: Literal["error", "warning"] = "error"
    detail: str
    field: str | None = None
    expected: str | None = None
    actual: str | None = None


class VerificationCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str | None = None


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    thread_id: str
    status: VerificationStatus
    attempts: int = 1
    checks: tuple[VerificationCheck, ...] = ()
    issues: tuple[VerificationIssue, ...] = ()
    independent_normalized_price: Decimal | None = None
    independent_acceptable_level: Decimal | None = None
    independent_order_value: Decimal | None = None
    independent_expected_label: str | None = None
    decision_label: str | None = None
    evidence_status: str | None = None
    provenance_complete: bool | None = None
    corrected: bool = False

    @property
    def passed(self) -> bool:
        return self.status in {
            VerificationStatus.PASSED,
            VerificationStatus.PASSED_AFTER_CORRECTION,
        }

    @property
    def error_issues(self) -> tuple[VerificationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")
