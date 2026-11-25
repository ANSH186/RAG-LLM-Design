"""Generate the full competition submission and validate its schema."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import get_environment
from src.evaluation.submission import validate_submission_file, write_submission
from src.pipeline import ProcurementPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--extraction-mode", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--submission", default=None, help="final submission path")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    env = get_environment(args.environment)
    threads_path = env.resolved(env.threads_file)
    template_path = env.resolved(env.sample_submission_file)
    frame = pd.read_csv(threads_path, dtype={"thread_id": str})
    if args.limit:
        frame = frame.head(args.limit)
    template = pd.read_csv(template_path, dtype={"thread_id": str})
    if args.limit and args.submission:
        raise SystemExit("--limit is for smoke tests; refusing to write a final submission from a subset")
    if len(frame) != len(template):
        # `--limit` is a development/smoke option; validate against the matching subset.
        subset = template[template["thread_id"].isin(set(frame["thread_id"]))]
        subset_path = ROOT / "output" / "eval" / f"template_subset_{len(frame)}.csv"
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(subset_path, index=False)
        template_for_run = subset_path
    else:
        template_for_run = template_path
    started = time.perf_counter()
    pipeline = None
    try:
        pipeline = ProcurementPipeline(
            environment_code=env.code,
            run_id=args.run_id,
            extraction_mode=args.extraction_mode,
        )
        print(f"run_id={pipeline.run_id} threads={len(frame)} output={pipeline.run_output_dir}")
        def progress(index: int, total: int) -> None:
            if index % 250 == 0 or index == total:
                print(f"processed {index}/{total} elapsed={time.perf_counter() - started:.1f}s")

        results = pipeline.run_dataframe(frame, progress=progress)
        labels = pipeline.labels_from_results(results)
        run_submission_path = pipeline.run_output_dir / "submission.csv"
        write_submission(labels, template_path=template_for_run, output_path=run_submission_path)
        validation = validate_submission_file(run_submission_path, template_path=template_for_run)
        metrics = pipeline.metrics.summary()
        (pipeline.run_output_dir / "metrics.json").write_text(
            metrics.model_dump_json(indent=2), encoding="utf-8"
        )
        (pipeline.run_output_dir / "labels.json").write_text(
            json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8"
        )
        if args.submission:
            target = Path(args.submission)
            if target.exists() and not args.overwrite:
                raise SystemExit(
                    f"refusing to overwrite existing submission {target}; pass --overwrite explicitly"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_submission_path, target)
            validation = validate_submission_file(target, template_path=template_for_run)
            print(f"wrote final submission {target}")
        print(json.dumps(validation, indent=2))
        print(json.dumps({
            "threads": metrics.total_threads,
            "labels": metrics.label_counts,
            "verification": metrics.verification_counts,
            "total_time_ms": metrics.total_time_ms,
            "avg_time_ms": metrics.avg_time_ms,
            "threads_per_second": metrics.threads_per_second,
        }, indent=2))
    finally:
        if pipeline is not None:
            pipeline.close()


if __name__ == "__main__":
    main()
