"""Evaluate the 40 labelled examples without label leakage.

Predictions are generated from sample_threads.csv and written to disk *before*
sample_answers.csv is opened.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import get_environment
from src.evaluation.evaluate import evaluate_predictions, write_evaluation_report
from src.pipeline import ProcurementPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    env = get_environment(args.environment)
    sample_threads_path = env.resolved(env.sample_threads_file)
    labels_path = env.resolved(env.sample_labels_file)
    # Intentionally only load the thread inputs here.
    threads = pd.read_csv(sample_threads_path, dtype={"thread_id": str})
    pipeline = ProcurementPipeline(environment_code=env.code, run_id=args.run_id)
    try:
        results = pipeline.run_dataframe(threads)
        labels = pipeline.labels_from_results(results)
        eval_dir = env.resolved(env.output_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = eval_dir / f"sample_predictions_{pipeline.run_id}.csv"
        prediction_frame = threads[["thread_id"]].copy()
        prediction_frame["label"] = prediction_frame["thread_id"].map(labels)
        prediction_frame.to_csv(predictions_path, index=False, lineterminator="\n")
        # Predictions are now on disk. Only now may expected labels be read.
        expected = pd.read_csv(labels_path, dtype={"thread_id": str})
        evaluation = evaluate_predictions(prediction_frame, expected)
        report_json = eval_dir / f"sample_evaluation_{pipeline.run_id}.json"
        report_json.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        report_md = eval_dir / f"sample_evaluation_{pipeline.run_id}.md"
        write_evaluation_report(evaluation, report_md, predictions_path=predictions_path)
        metrics = pipeline.metrics.summary()
        (eval_dir / f"sample_metrics_{pipeline.run_id}.json").write_text(
            metrics.model_dump_json(indent=2), encoding="utf-8"
        )
        print(json.dumps({
            "predictions": str(predictions_path),
            "report": str(report_md),
            "accuracy": evaluation["exact_match_accuracy"],
            "correct": evaluation["exact_match_count"],
            "threads": evaluation["thread_count"],
            "errors": evaluation["errors"],
        }, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
