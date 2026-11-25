"""Submission writing and strict schema validation."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.errors import ProcurementError

LABEL_RE = re.compile(
    r"^(?:"
    r"ACCEPT:ACCEPTABLE_PRICE:|"
    r"REJECT:ABOVE_WALKAWAY:|"
    r"REJECT:BLACKLISTED_SUPPLIER:|"
    r"ESCALATE:MISSING_INFO:|"
    r"ESCALATE:PROMPT_INJECTION:|"
    r"ESCALATE:BANNED_TERM:|"
    r"ESCALATE:OVER_APPROVAL_LIMIT:|"
    r"COUNTER:COUNTER_OFFER:\d+)"
)


class SubmissionValidationError(ProcurementError):
    pass


def write_submission(
    labels: dict[str, str],
    *,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    template = pd.read_csv(template_path, dtype={"thread_id": str})
    if list(template.columns) != ["thread_id", "label"]:
        raise SubmissionValidationError(f"template columns must be thread_id,label, got {list(template.columns)}")
    missing = [thread_id for thread_id in template["thread_id"] if thread_id not in labels]
    extra = [thread_id for thread_id in labels if thread_id not in set(template["thread_id"])]
    if missing or extra:
        raise SubmissionValidationError(
            f"submission thread coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    ordered = pd.DataFrame(
        {"thread_id": template["thread_id"], "label": [labels[thread_id] for thread_id in template["thread_id"]]}
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(output_path, index=False, lineterminator="\n", encoding="utf-8")
    validate_submission_file(output_path, template_path=template_path)
    return output_path


def validate_submission_file(
    path: str | Path,
    *,
    template_path: str | Path,
) -> dict:
    path = Path(path)
    template = pd.read_csv(template_path, dtype={"thread_id": str})
    submission = pd.read_csv(path, dtype={"thread_id": str}, keep_default_na=False)
    problems: list[str] = []
    if list(submission.columns) != ["thread_id", "label"]:
        problems.append(f"columns must be ['thread_id','label'], got {list(submission.columns)}")
    if len(submission) != len(template):
        problems.append(f"row count {len(submission)} != template {len(template)}")
    if submission["thread_id"].duplicated().any():
        problems.append("duplicate thread_id rows")
    if set(submission["thread_id"]) != set(template["thread_id"]):
        problems.append("thread_id set does not match template")
    for index, label in enumerate(submission["label"].astype(str)):
        if not LABEL_RE.fullmatch(label):
            problems.append(f"invalid label at row {index}: {label!r}")
            if len(problems) > 10:
                break
    if problems:
        raise SubmissionValidationError("; ".join(problems))
    return {
        "path": str(path),
        "rows": len(submission),
        "columns": list(submission.columns),
        "unique_thread_ids": int(submission["thread_id"].nunique()),
        "valid_labels": True,
    }
