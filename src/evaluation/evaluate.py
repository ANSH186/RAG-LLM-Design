"""Sample evaluation with strict separation of prediction and label loading."""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.evaluation.submission import LABEL_RE


def _action(label: str) -> str:
    return label.split(":", 1)[0]


def _reason(label: str) -> str:
    parts = label.split(":")
    return parts[1] if len(parts) > 1 else ""


def evaluate_predictions(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict:
    if list(predictions.columns) != ["thread_id", "label"]:
        raise ValueError("predictions must have thread_id,label")
    if list(labels.columns) != ["thread_id", "label"]:
        raise ValueError("labels must have thread_id,label")
    merged = predictions.merge(labels, on="thread_id", how="inner", suffixes=("_pred", "_expected"))
    if len(merged) != len(labels):
        missing = set(labels["thread_id"]) - set(predictions["thread_id"])
        raise ValueError(f"predictions missing {len(missing)} labelled threads")
    merged["correct"] = merged["label_pred"] == merged["label_expected"]
    actions = sorted(set(merged["label_expected"].map(_action)) | set(merged["label_pred"].map(_action)))
    per_action = {}
    for action in actions:
        subset = merged[merged["label_expected"].map(_action) == action]
        per_action[action] = {
            "count": len(subset),
            "correct": int(subset["correct"].sum()),
            "accuracy": round(float(subset["correct"].mean()), 6) if len(subset) else None,
        }
    confusion = Counter(zip(merged["label_expected"], merged["label_pred"]))
    errors = merged[~merged["correct"]][["thread_id", "label_expected", "label_pred"]].to_dict("records")
    exact = int(merged["correct"].sum())
    return {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "thread_count": len(merged),
        "exact_match_count": exact,
        "exact_match_accuracy": round(exact / len(merged), 6) if len(merged) else None,
        "per_action": per_action,
        "confusion_matrix": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(confusion.items())
        ],
        "errors": errors,
        "label_format_valid": all(
            LABEL_RE.fullmatch(str(label)) for label in predictions["label"].astype(str)
        ),
    }


def write_evaluation_report(evaluation: dict, path: str | Path, *, predictions_path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sample Evaluation",
        "",
        f"- Predictions: `{predictions_path}`",
        f"- Evaluated at: {evaluation['evaluated_at']}",
        f"- Threads: {evaluation['thread_count']}",
        f"- Exact-match correct: {evaluation['exact_match_count']}",
        f"- Exact-match accuracy: {evaluation['exact_match_accuracy']}",
        f"- Label format valid: {evaluation['label_format_valid']}",
        "",
        "## Per action",
        "",
        "| Action | Count | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for action, stats in evaluation["per_action"].items():
        lines.append(f"| {action} | {stats['count']} | {stats['correct']} | {stats['accuracy']} |")
    lines += [
        "",
        "## Errors",
        "",
    ]
    if evaluation["errors"]:
        lines += ["| Thread | Expected | Predicted |", "|---|---|---|"]
        for error in evaluation["errors"]:
            lines.append(f"| {error['thread_id']} | {error['label_expected']} | {error['label_pred']} |")
    else:
        lines.append("No errors.")
    lines += [
        "",
        "## Integrity statement",
        "",
        "Predictions were generated and written before the label file was opened. The prediction "
        "pipeline never received labels or thread-specific expected answers.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
