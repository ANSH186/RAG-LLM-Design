"""Content-addressed cache for LLM extraction responses."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import PROJECT_ROOT
from src.utils.hashing import sha256_json


class LLMCache:
    def __init__(self, directory: str | Path | None = None) -> None:
        base = Path(directory) if directory else PROJECT_ROOT / "output" / "cache" / "llm"
        base.mkdir(parents=True, exist_ok=True)
        self.directory = base

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, *, model: str, system_prompt: str, user_prompt: str) -> dict | None:
        key = sha256_json({"model": model, "system": system_prompt, "user": user_prompt})
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, *, model: str, system_prompt: str, user_prompt: str, value: dict) -> None:
        key = sha256_json({"model": model, "system": system_prompt, "user": user_prompt})
        self._path(key).write_text(json.dumps(value, indent=2), encoding="utf-8")
