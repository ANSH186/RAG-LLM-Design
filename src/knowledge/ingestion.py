"""Knowledge ingestion orchestration and artifact persistence."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.config import EnvironmentConfig, ensure_runtime_dirs
from src.errors import KnowledgeIngestionError
from src.knowledge.policy_compiler import compile_knowledge, validate_compiled_knowledge
from src.knowledge.schema import CompiledKnowledge
from src.utils.hashing import sha256_text


class IngestionReport:
    def __init__(self, knowledge: CompiledKnowledge, artifact_path: Path, manifest_path: Path) -> None:
        self.knowledge = knowledge
        self.artifact_path = artifact_path
        self.manifest_path = manifest_path

    def summary(self) -> dict:
        return {
            "environment": self.knowledge.environment,
            "categories": len(self.knowledge.categories),
            "suppliers": len(self.knowledge.suppliers),
            "amendments": len(self.knowledge.amendments),
            "precedents": len(self.knowledge.precedents),
            "documents": len(self.knowledge.documents),
            "configuration_hash": self.knowledge.configuration_hash,
            "artifact": str(self.artifact_path),
        }


def ingest_knowledge(env: EnvironmentConfig | None = None) -> IngestionReport:
    from src.config import get_environment

    env = env or get_environment()
    ensure_runtime_dirs(env)
    root = env.resolved(env.knowledge_root)
    knowledge = compile_knowledge(knowledge_root=root, environment=env.code)
    validate_compiled_knowledge(knowledge)

    structured_dir = env.resolved(env.structured_dir)
    structured_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = structured_dir / "knowledge.json"
    artifact_path.write_text(
        knowledge.model_dump_json(indent=2),
        encoding="utf-8",
    )
    manifest = {
        "environment": env.code,
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration_hash": knowledge.configuration_hash,
        "source_root": knowledge.source_root,
        "counts": {
            "documents": len(knowledge.documents),
            "categories": len(knowledge.categories),
            "suppliers": len(knowledge.suppliers),
            "amendments": len(knowledge.amendments),
            "precedents": len(knowledge.precedents),
        },
        "documents": [
            {
                "document_id": doc.document_id,
                "document_type": doc.document_type.value,
                "source_file": doc.source_file,
                "content_hash": doc.content_hash,
                "effective_from": doc.effective_from.isoformat() if doc.effective_from else None,
                "category": doc.category,
                "supplier": doc.supplier,
                "amendment_id": doc.amendment_id,
            }
            for doc in knowledge.documents
        ],
    }
    manifest_path = structured_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return IngestionReport(knowledge, artifact_path, manifest_path)


def load_knowledge(env: EnvironmentConfig | None = None) -> CompiledKnowledge:
    from src.config import get_environment

    env = env or get_environment()
    path = env.resolved(env.structured_dir) / "knowledge.json"
    if not path.exists():
        raise KnowledgeIngestionError(
            f"compiled knowledge not found at {path}; run `python scripts/build_knowledge.py`"
        )
    return CompiledKnowledge.model_validate_json(path.read_text(encoding="utf-8"))


def verify_source_hashes(knowledge: CompiledKnowledge, project_root: Path) -> list[str]:
    """Return documents whose source Markdown changed after compilation."""
    problems: list[str] = []
    for doc in knowledge.documents:
        # source_file is relative to the configured knowledge root.
        path = Path(knowledge.source_root) / doc.source_file
        if not path.exists():
            problems.append(f"{doc.document_id}: missing source file {path}")
            continue
        current = sha256_text(path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n"))
        if current != doc.content_hash:
            problems.append(f"{doc.document_id}: source hash changed")
    return problems
