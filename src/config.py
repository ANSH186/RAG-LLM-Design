"""Application configuration.

All official procurement values (targets, walk-aways, GST rates, limits, statuses,
contract overrides and amendments) come from the knowledge base, never from this file.
This module only holds infrastructure and execution settings.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class PathSettings(BaseModel):
    environments: str = "config/environments"
    knowledge_structured: str = "knowledge_base/structured"
    knowledge_indexes: str = "knowledge_base/indexes"
    knowledge_snapshots: str = "knowledge_base/snapshots"
    output: str = "output"
    reports: str = "reports"
    audit: str = "output/audit"


class DecisionSettings(BaseModel):
    max_verification_attempts: int = Field(default=2, ge=1, le=3)
    display_tolerance: str = "0.000001"


class RetrievalSettings(BaseModel):
    top_k: int = Field(default=8, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1)
    bm25_weight: float = Field(default=1.0, ge=0)
    dense_weight: float = Field(default=1.0, ge=0)
    use_reranker: bool = True
    rerank_top_n: int = Field(default=12, ge=1, le=50)
    embedding_backend: str = "hash"
    embedding_dim: int = Field(default=512, ge=64, le=4096)
    max_context_chunks: int = Field(default=10, ge=1, le=50)


class ObservabilitySettings(BaseModel):
    log_level: str = "INFO"
    json_logs: bool = True
    trace_directory: str = "reports/traces"


class Settings(BaseModel):
    project_name: str
    default_environment: str
    paths: PathSettings
    decision: DecisionSettings
    retrieval: RetrievalSettings
    observability: ObservabilitySettings


class EnvironmentConfig(BaseModel):
    name: str
    code: str
    knowledge_root: str
    threads_file: str
    sample_threads_file: str
    sample_labels_file: str
    sample_submission_file: str
    structured_dir: str
    index_dir: str
    snapshot_dir: str
    output_dir: str
    audit_dir: str
    retrieval_eval_file: str | None = None

    def resolved(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (PROJECT_ROOT / path)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings(**_read_yaml(PROJECT_ROOT / "config" / "settings.yaml"))
    # Environment variables are operational overrides, not policy data.
    env = os.getenv("PROCUREMENT_ENV")
    if env:
        settings.default_environment = env
    if os.getenv("RETRIEVAL_TOP_K"):
        settings.retrieval.top_k = int(os.environ["RETRIEVAL_TOP_K"])
    if os.getenv("RETRIEVAL_RRF_K"):
        settings.retrieval.rrf_k = int(os.environ["RETRIEVAL_RRF_K"])
    if os.getenv("RETRIEVAL_USE_RERANKER"):
        settings.retrieval.use_reranker = os.environ["RETRIEVAL_USE_RERANKER"].lower() in {"1", "true", "yes"}
    if os.getenv("RETRIEVAL_EMBEDDING_BACKEND"):
        settings.retrieval.embedding_backend = os.environ["RETRIEVAL_EMBEDDING_BACKEND"]
    if os.getenv("RETRIEVAL_EMBEDDING_DIM"):
        settings.retrieval.embedding_dim = int(os.environ["RETRIEVAL_EMBEDDING_DIM"])
    if os.getenv("LOG_LEVEL"):
        settings.observability.log_level = os.environ["LOG_LEVEL"]
    return settings


@lru_cache(maxsize=8)
def get_environment(code: str | None = None) -> EnvironmentConfig:
    settings = get_settings()
    env_code = code or settings.default_environment
    path = PROJECT_ROOT / settings.paths.environments / f"{env_code}.yaml"
    return EnvironmentConfig(**_read_yaml(path))


def project_path(relative: str | Path) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_runtime_dirs(env: EnvironmentConfig | None = None) -> None:
    settings = get_settings()
    env = env or get_environment()
    for path in (
        project_path(settings.paths.knowledge_structured),
        project_path(settings.paths.knowledge_indexes),
        project_path(settings.paths.knowledge_snapshots),
        env.resolved(env.structured_dir),
        env.resolved(env.index_dir),
        env.resolved(env.snapshot_dir),
        env.resolved(env.output_dir),
        env.resolved(env.audit_dir),
        project_path(settings.paths.reports),
    ):
        path.mkdir(parents=True, exist_ok=True)
