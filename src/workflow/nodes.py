"""LangGraph workflow nodes."""
from __future__ import annotations

import time
from datetime import UTC, datetime

from src.audit import AuditRecord
from src.decision.decision_engine import decide
from src.decision.security_rules import scan_security
from src.evidence.provenance import build_provenance_map
from src.extraction.deterministic_extractor import DeterministicExtractor
from src.extraction.extraction_schema import ExtractionMethod
from src.extraction.normalizer import OfferNormalizer
from src.extraction.validation import validate_extraction_grounding
from src.knowledge.temporal_resolver import resolve_policy_snapshot
from src.observability.metrics import ThreadMetrics
from src.verification.decision_verifier import verify_decision
from src.verification.evidence_verifier import verify_evidence_package
from src.workflow.state import (
    PipelineServices,
    ProcurementState,
    factor_resolver,
    make_orchestrator,
)


def _merge_warnings(state: ProcurementState, *values) -> tuple[str, ...]:
    collected: list[str] = list(state.get("warnings", ()))
    for value in values:
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            collected.extend(str(v) for v in value)
        else:
            collected.append(str(value))
    return tuple(dict.fromkeys(v for v in collected if v))


def _merge_errors(state: ProcurementState, *values) -> tuple[str, ...]:
    collected: list[str] = list(state.get("errors", ()))
    for value in values:
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            collected.extend(str(v) for v in value)
        else:
            collected.append(str(value))
    return tuple(dict.fromkeys(v for v in collected if v))


class WorkflowNodes:
    def __init__(self, services: PipelineServices) -> None:
        self.services = services

    def _timed(self, name: str):
        def wrapper(fn):
            def node(state: ProcurementState):
                started = time.perf_counter()
                with self.services.tracer.span(
                    name, {"thread_id": state.get("thread_id"), "run_id": self.services.run_id}
                ):
                    update = fn(state) or {}
                elapsed = round((time.perf_counter() - started) * 1000.0, 3)
                timings = dict(state.get("timings_ms", {}))
                timings[name] = elapsed
                update["timings_ms"] = timings
                update["stage_trace"] = tuple(state.get("stage_trace", ())) + (name,)
                return update

            return node

        return wrapper

    # ------------------------------------------------------------------ nodes
    def load_thread(self, state: ProcurementState) -> dict:
        row = state.get("thread_row") or {}
        errors: list[str] = []
        try:
            thread_id = str(row["thread_id"])
            category = str(row["category"])
            supplier = str(row["supplier"])
            date_text = str(row["date"])
            quantity = int(row["quantity"])
            round_number = int(row["round"])
            message = str(row["supplier_message"])
            datetime.strptime(date_text, "%Y-%m-%d")
            if not message.strip():
                errors.append("empty supplier message")
            if quantity <= 0:
                errors.append("quantity must be positive")
            if round_number <= 0:
                errors.append("round must be positive")
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "errors": _merge_errors(state, f"invalid thread row: {exc}"),
                "safe_failure": True,
                "final_label": "ESCALATE:MISSING_INFO:",
                "attempts": 1,
            }
        if errors:
            return {
                "errors": _merge_errors(state, errors),
                "safe_failure": True,
                "final_label": "ESCALATE:MISSING_INFO:",
                "attempts": 1,
            }
        return {
            "thread_id": thread_id,
            "category_slug": category,
            "supplier_name": supplier,
            "negotiation_date": date_text,
            "quantity": quantity,
            "round_number": round_number,
            "our_last_counter": (
                None if row.get("our_last_counter") is None else float(row["our_last_counter"])
            ),
            "attempts": 1,
            "warnings": tuple(state.get("warnings", ())),
            "errors": tuple(state.get("errors", ())),
        }

    def resolve_policy(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        try:
            snapshot = resolve_policy_snapshot(
                self.services.knowledge,
                thread_id=state["thread_id"],
                category_slug=state["category_slug"],
                supplier_name=state["supplier_name"],
                negotiation_date=state["negotiation_date"],
            )
        except Exception as exc:  # noqa: BLE001 - any policy failure routes to audited safe escalation
            return {
                "errors": _merge_errors(state, f"policy resolution failed: {exc}"),
                "safe_failure": True,
                "final_label": "ESCALATE:MISSING_INFO:",
            }
        return {"snapshot": snapshot}

    def understand_thread(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        snapshot = state["snapshot"]
        resolver = factor_resolver(self.services, state["category_slug"])
        message = str(state["thread_row"]["supplier_message"])
        injections, banned = scan_security(message)
        extractor = DeterministicExtractor(category_factor_resolver=resolver)
        extraction = extractor.extract(
            thread_id=state["thread_id"], source_message=message, snapshot=snapshot
        )
        normalized = OfferNormalizer(snapshot, factor_resolver=resolver).normalize(
            extraction, quantity=int(state["quantity"])
        )
        early_rule = None
        if not normalized.is_usable:
            early_rule = "missing_information"
        elif snapshot.supplier_status.value == "BLACKLISTED":
            early_rule = "blacklisted_supplier"
        elif injections:
            early_rule = "prompt_injection"
        elif banned:
            early_rule = "banned_term"
        return {
            "pre_extraction": extraction,
            "pre_normalized": normalized,
            "security_injections": tuple(injections),
            "security_banned": tuple(banned),
            "early_rule": early_rule,
            "warnings": _merge_warnings(state, normalized.warnings),
        }

    def construct_query(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        category = self.services.knowledge.categories[state["category_slug"]]
        offer_text = state["pre_extraction"].offer_text or ""
        query = (
            f"{category.title}; normalized unit {category.normalized_unit}; "
            f"supplier {state['supplier_name']}; negotiation date {state['negotiation_date']}; "
            f"round {state['round_number']} quantity {state['quantity']}; "
            f"supplier status and contract; amendments effective on or before the date; "
            f"target price walk-away price approval limit concession schedule; "
            f"banned terms security prompt injection currency GST normalization. "
            f"Latest supplier sentence: {offer_text}"
        )
        return {"retrieval_query": query}

    def hybrid_retrieve(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        snapshot = state["snapshot"]
        query = state.get("retrieval_query") or ""
        from datetime import date

        from src.retrieval.schemas import MetadataFilter

        filters = MetadataFilter(
            category=snapshot.category_slug,
            supplier=snapshot.supplier_name,
            negotiation_date=date.fromisoformat(snapshot.negotiation_date.isoformat()),
        )
        result = self.services.retriever.retrieve(
            query,
            filters=filters,
            mandatory_document_ids=self.services.retriever.mandatory_document_ids(snapshot),
        )
        warnings = _merge_warnings(state)
        # Top-k omission is not a missing-evidence condition when the source is present
        # in the registry: the evidence layer injects mandatory documents directly.
        # Warn only if a mandatory document is genuinely absent from the knowledge source.
        available_documents = set(self.services.registry.documents)
        truly_missing = [doc_id for doc_id in result.missing_evidence if doc_id not in available_documents]
        if truly_missing:
            warnings = tuple(
                list(warnings)
                + [f"mandatory source absent from knowledge registry: {', '.join(truly_missing)}"]
            )
        return {"retrieval": result, "warnings": tuple(dict.fromkeys(warnings))}

    def build_evidence(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        package = self.services.context_builder.build(
            thread_id=state["thread_id"],
            source_message=str(state["thread_row"]["supplier_message"]),
            snapshot=state["snapshot"],
            retrieval=state.get("retrieval"),
            injection_findings=state.get("security_injections", ()),
            banned_term_findings=state.get("security_banned", ()),
        )
        return {
            "evidence": package,
            "warnings": _merge_warnings(state, package.warnings),
        }

    def validate_evidence(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        package = state.get("evidence")
        if package is None:
            return {
                "errors": _merge_errors(state, "evidence package missing"),
                "safe_failure": True,
                "final_label": "ESCALATE:MISSING_INFO:",
            }
        issues, checks, provenance_complete = verify_evidence_package(
            package=package,
            extraction=state["pre_extraction"],
            knowledge=self.services.knowledge,
            registry=self.services.registry,
        )
        error_issues = [issue for issue in issues if issue.severity == "error"]
        validation = {
            "passed": not error_issues,
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "checks": [check.model_dump(mode="json") for check in checks],
            "provenance_complete": provenance_complete,
        }
        if error_issues:
            return {
                "evidence_validation": validation,
                "errors": _merge_errors(state, [issue.detail for issue in error_issues]),
                "safe_failure": True,
                "final_label": "ESCALATE:MISSING_INFO:",
            }
        return {"evidence_validation": validation}

    def build_context(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        package = state.get("evidence")
        if package is None or not package.context_text:
            return {
                "errors": _merge_errors(state, "context construction failed"),
                "safe_failure": True,
                "final_label": "ESCALATE:MISSING_INFO:",
            }
        return {"context_hash": package.context_hash}

    def extract_offer(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        if state.get("early_rule") or self.services.extraction_mode in {"deterministic", "det", "off"}:
            extraction = state["pre_extraction"]
            normalized = state["pre_normalized"]
            outcome = {
                "requested_mode": self.services.extraction_mode,
                "used_method": ExtractionMethod.DETERMINISTIC.value,
                "fallback_used": False,
                "llm_calls": 0,
                "warnings": list(extraction.warnings),
            }
        else:
            orchestrator = make_orchestrator(self.services, state["category_slug"])
            outcome_model = orchestrator.extract(
                thread_id=state["thread_id"],
                source_message=str(state["thread_row"]["supplier_message"]),
                snapshot=state["snapshot"],
            )
            extraction = outcome_model.extraction
            normalized = OfferNormalizer(
                state["snapshot"], factor_resolver=factor_resolver(self.services, state["category_slug"])
            ).normalize(extraction, quantity=int(state["quantity"]))
            outcome = outcome_model.model_dump(mode="json")
        return {
            "extraction": extraction,
            "normalized": normalized,
            "extraction_outcome": outcome,
            "warnings": _merge_warnings(state, outcome.get("warnings")),
        }

    def validate_extraction(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        validation = validate_extraction_grounding(state["extraction"], state["snapshot"])
        if not validation.valid:
            return {
                "errors": _merge_errors(state, validation.errors),
                "safe_failure": True,
                "final_label": "ESCALATE:MISSING_INFO:",
            }
        return {"warnings": _merge_warnings(state, validation.warnings)}

    def normalize_offer(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        normalized = OfferNormalizer(
            state["snapshot"], factor_resolver=factor_resolver(self.services, state["category_slug"])
        ).normalize(state["extraction"], quantity=int(state["quantity"]))
        return {
            "normalized": normalized,
            "warnings": _merge_warnings(state, normalized.warnings),
        }

    def deterministic_decision(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        decision = decide(
            thread_id=state["thread_id"],
            category=state["category_slug"],
            supplier=state["supplier_name"],
            round_number=int(state["round_number"]),
            quantity=int(state["quantity"]),
            extraction=state["extraction"],
            normalized=state["normalized"],
            snapshot=state["snapshot"],
            injection_findings=list(state.get("security_injections", ())),
            banned_term_findings=list(state.get("security_banned", ())),
        )
        return {
            "decision": decision,
            "warnings": _merge_warnings(state, decision.warnings),
        }

    def independent_verification(self, state: ProcurementState) -> dict:
        if state.get("safe_failure"):
            return {}
        attempt = int(state.get("attempts", 1))
        verification = verify_decision(
            decision=state["decision"],
            extraction=state["extraction"],
            normalized=state["normalized"],
            snapshot=state["snapshot"],
            knowledge=self.services.knowledge,
            evidence_package=state.get("evidence"),
            registry=self.services.registry,
            attempt=attempt,
            corrected=attempt > 1 and state.get("decision") is not None,
        )
        history = tuple(state.get("verification_history", ())) + (verification.model_dump(mode="json"),)
        return {"verification": verification, "verification_history": history}

    def bounded_recheck(self, state: ProcurementState) -> dict:
        attempts = int(state.get("attempts", 1)) + 1
        snapshot = state["snapshot"]
        resolver = factor_resolver(self.services, state["category_slug"])
        extraction = DeterministicExtractor(category_factor_resolver=resolver).extract(
            thread_id=state["thread_id"],
            source_message=str(state["thread_row"]["supplier_message"]),
            snapshot=snapshot,
        )
        normalized = OfferNormalizer(snapshot, factor_resolver=resolver).normalize(
            extraction, quantity=int(state["quantity"])
        )
        decision = decide(
            thread_id=state["thread_id"],
            category=state["category_slug"],
            supplier=state["supplier_name"],
            round_number=int(state["round_number"]),
            quantity=int(state["quantity"]),
            extraction=extraction,
            normalized=normalized,
            snapshot=snapshot,
            injection_findings=list(state.get("security_injections", ())),
            banned_term_findings=list(state.get("security_banned", ())),
        )
        return {
            "attempts": attempts,
            "extraction": extraction,
            "normalized": normalized,
            "decision": decision,
            "warnings": _merge_warnings(
                state, "bounded deterministic re-evaluation performed after verification failure"
            ),
        }

    def safe_failure(self, state: ProcurementState) -> dict:
        errors = _merge_errors(state)
        verification = state.get("verification")
        if verification and not verification.passed:
            errors = _merge_errors(
                state,
                [f"verification:{issue.code}:{issue.detail}" for issue in verification.error_issues],
            )
        return {
            "safe_failure": True,
            "final_label": "ESCALATE:MISSING_INFO:",
            "errors": errors,
        }

    def persist_audit(self, state: ProcurementState) -> dict:
        if state.get("audit_persisted"):
            return {}
        decision = state.get("decision")
        verification = state.get("verification")
        final_label = state.get("final_label") or (
            decision.label if decision is not None else "ESCALATE:MISSING_INFO:"
        )
        package = state.get("evidence")
        evidence_payload = None
        if package is not None:
            provenance = build_provenance_map(package.policy_snapshot, package)
            evidence_payload = {
                "snapshot_id": package.snapshot_id,
                "context_hash": package.context_hash,
                "context_char_count": len(package.context_text),
                "evidence_count": package.evidence_count,
                "trust_boundary": package.trust_boundary,
                "mandatory_document_ids": list(package.mandatory_document_ids),
                "mandatory_missing": list(package.mandatory_missing),
                "official_evidence": [
                    item.model_dump(mode="json") for item in package.official_evidence
                ],
                "retrieved_evidence": [
                    item.model_dump(mode="json") for item in package.retrieved_evidence
                ],
                "precedents": [item.model_dump(mode="json") for item in package.precedents],
                "supplier_message": package.supplier_message.model_dump(mode="json"),
                "provenance": provenance.model_dump(mode="json"),
            }
        timings = dict(state.get("timings_ms", {}))
        thread_started = state.get("thread_started_perf")
        if thread_started:
            timings["total"] = round((time.perf_counter() - thread_started) * 1000.0, 3)
        record = AuditRecord(
            run_id=self.services.run_id,
            thread_id=state.get("thread_id") or str((state.get("thread_row") or {}).get("thread_id", "unknown")),
            created_at=datetime.now(UTC).isoformat(),
            input={
                "date": state.get("negotiation_date"),
                "category": state.get("category_slug"),
                "supplier": state.get("supplier_name"),
                "quantity": state.get("quantity"),
                "round": state.get("round_number"),
                "our_last_counter": state.get("our_last_counter"),
                "supplier_message": (state.get("thread_row") or {}).get("supplier_message"),
                "early_rule_route": state.get("early_rule"),
            },
            final_label=final_label,
            safe_failure=bool(state.get("safe_failure")),
            decision=decision.model_dump(mode="json") if decision is not None else None,
            extraction=state.get("extraction").model_dump(mode="json") if state.get("extraction") else None,
            normalization=state.get("normalized").model_dump(mode="json") if state.get("normalized") else None,
            policy_snapshot=state.get("snapshot").model_dump(mode="json") if state.get("snapshot") else None,
            evidence=evidence_payload,
            verification=verification.model_dump(mode="json") if verification is not None else None,
            security={
                "injection_findings": [
                    f.model_dump(mode="json") for f in state.get("security_injections", ())
                ],
                "banned_term_findings": [
                    f.model_dump(mode="json") for f in state.get("security_banned", ())
                ],
            },
            timings_ms=timings,
            warnings=tuple(state.get("warnings", ())),
            errors=tuple(state.get("errors", ())),
            stage_trace=tuple(state.get("stage_trace", ())),
        )
        self.services.audit_store.write(record)
        extraction = state.get("extraction")
        self.services.metrics.add(
            ThreadMetrics(
                thread_id=record.thread_id,
                label=final_label,
                applied_rule=decision.applied_rule.value if decision is not None else None,
                verification_status=verification.status.value if verification is not None else None,
                extraction_method=extraction.extraction_method.value if extraction is not None else None,
                extraction_confidence=extraction.extraction_confidence if extraction is not None else None,
                retrieval_confidence=(
                    state.get("retrieval").confidence if state.get("retrieval") is not None else None
                ),
                evidence_count=package.evidence_count if package is not None else 0,
                llm_calls=int((state.get("extraction_outcome") or {}).get("llm_calls") or 0),
                timings_ms=timings,
                warnings=tuple(state.get("warnings", ())),
                failures=tuple(state.get("errors", ())),
                stage_trace=tuple(state.get("stage_trace", ())),
            )
        )
        return {"audit_persisted": True, "final_label": final_label}

    def finalize_output(self, state: ProcurementState) -> dict:
        return {
            "final_label": state.get("final_label")
            or (
                state.get("decision").label
                if state.get("decision") is not None
                else "ESCALATE:MISSING_INFO:"
            )
        }

    def build_node_map(self) -> dict:
        mapping = {
            "load_thread": self.load_thread,
            "resolve_policy": self.resolve_policy,
            "understand_thread": self.understand_thread,
            "construct_query": self.construct_query,
            "hybrid_retrieve": self.hybrid_retrieve,
            "build_evidence": self.build_evidence,
            "validate_evidence": self.validate_evidence,
            "build_context": self.build_context,
            "extract_offer": self.extract_offer,
            "validate_extraction": self.validate_extraction,
            "normalize_offer": self.normalize_offer,
            "deterministic_decision": self.deterministic_decision,
            "independent_verification": self.independent_verification,
            "bounded_recheck": self.bounded_recheck,
            "safe_failure": self.safe_failure,
            "persist_audit": self.persist_audit,
            "finalize_output": self.finalize_output,
        }
        return {name: self._timed(name)(fn) for name, fn in mapping.items()}
