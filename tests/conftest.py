from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evidence.context_builder import ContextBuilder
from src.evidence.source_registry import SourceRegistry
from src.knowledge.policy_compiler import compile_knowledge, validate_compiled_knowledge
from src.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
ACME_ROOT = ROOT / "data" / "raw" / "playbook" / "playbook"


@pytest.fixture(scope="session")
def knowledge():
    value = compile_knowledge(knowledge_root=ACME_ROOT, environment="acme")
    validate_compiled_knowledge(value)
    return value


@pytest.fixture(scope="session")
def retriever(knowledge):
    return HybridRetriever(knowledge)


@pytest.fixture(scope="session")
def registry(knowledge):
    return SourceRegistry(knowledge)


@pytest.fixture(scope="session")
def context_builder(registry):
    return ContextBuilder(registry)


@pytest.fixture(scope="session")
def sample_threads():
    return pd.read_csv(ROOT / "data" / "raw" / "sample_threads.csv", dtype={"thread_id": str})


@pytest.fixture(scope="session")
def sample_answers():
    return pd.read_csv(ROOT / "data" / "raw" / "sample_answers.csv", dtype={"thread_id": str})


@pytest.fixture(scope="session")
def all_threads():
    return pd.read_csv(ROOT / "data" / "raw" / "threads.csv", dtype={"thread_id": str})
