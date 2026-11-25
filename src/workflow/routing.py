"""Conditional routing for the procurement workflow."""
from __future__ import annotations

from src.workflow.state import ProcurementState


def route_after_load(state: ProcurementState) -> str:
    return "safe_failure" if state.get("safe_failure") else "resolve_policy"


def route_after_resolve(state: ProcurementState) -> str:
    return "safe_failure" if state.get("safe_failure") else "understand_thread"


def route_after_understanding(state: ProcurementState) -> str:
    if state.get("safe_failure"):
        return "safe_failure"
    # Missing information, blacklist, injection and banned terms are decisive before
    # price evidence is needed. Skip expensive retrieval but still load mandatory policy.
    if state.get("early_rule"):
        return "build_evidence"
    return "construct_query"


def route_after_evidence_validation(state: ProcurementState) -> str:
    return "safe_failure" if state.get("safe_failure") else "build_context"


def route_after_extraction_validation(state: ProcurementState) -> str:
    return "safe_failure" if state.get("safe_failure") else "normalize_offer"


def route_after_verification(state: ProcurementState, max_attempts: int = 2) -> str:
    verification = state.get("verification")
    if verification is not None and verification.passed:
        return "persist_audit"
    if int(state.get("attempts", 1)) < max_attempts:
        return "bounded_recheck"
    return "safe_failure"
