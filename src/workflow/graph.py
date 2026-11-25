"""Compile the explicit LangGraph workflow."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.workflow.nodes import WorkflowNodes
from src.workflow.routing import (
    route_after_evidence_validation,
    route_after_extraction_validation,
    route_after_load,
    route_after_resolve,
    route_after_understanding,
    route_after_verification,
)
from src.workflow.state import PipelineServices, ProcurementState


def build_graph(services: PipelineServices):
    nodes = WorkflowNodes(services).build_node_map()
    graph = StateGraph(ProcurementState)
    for name, function in nodes.items():
        graph.add_node(name, function)

    graph.add_edge(START, "load_thread")
    graph.add_conditional_edges(
        "load_thread",
        route_after_load,
        {"resolve_policy": "resolve_policy", "safe_failure": "safe_failure"},
    )
    graph.add_conditional_edges(
        "resolve_policy",
        route_after_resolve,
        {"understand_thread": "understand_thread", "safe_failure": "safe_failure"},
    )
    graph.add_conditional_edges(
        "understand_thread",
        route_after_understanding,
        {
            "construct_query": "construct_query",
            "build_evidence": "build_evidence",
            "safe_failure": "safe_failure",
        },
    )
    graph.add_edge("construct_query", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "build_evidence")
    graph.add_edge("build_evidence", "validate_evidence")
    graph.add_conditional_edges(
        "validate_evidence",
        route_after_evidence_validation,
        {"build_context": "build_context", "safe_failure": "safe_failure"},
    )
    graph.add_edge("build_context", "extract_offer")
    graph.add_edge("extract_offer", "validate_extraction")
    graph.add_conditional_edges(
        "validate_extraction",
        route_after_extraction_validation,
        {"normalize_offer": "normalize_offer", "safe_failure": "safe_failure"},
    )
    graph.add_edge("normalize_offer", "deterministic_decision")
    graph.add_edge("deterministic_decision", "independent_verification")
    max_attempts = get_settings().decision.max_verification_attempts
    graph.add_conditional_edges(
        "independent_verification",
        lambda state: route_after_verification(state, max_attempts),
        {
            "persist_audit": "persist_audit",
            "bounded_recheck": "bounded_recheck",
            "safe_failure": "safe_failure",
        },
    )
    graph.add_edge("bounded_recheck", "independent_verification")
    graph.add_edge("safe_failure", "persist_audit")
    graph.add_edge("persist_audit", "finalize_output")
    graph.add_edge("finalize_output", END)
    return graph.compile()
