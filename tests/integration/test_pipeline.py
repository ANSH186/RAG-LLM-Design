import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.submission import (
    SubmissionValidationError,
    validate_submission_file,
    write_submission,
)
from src.pipeline import ProcurementPipeline

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def pipeline(tmp_path):
    instance = ProcurementPipeline(
        run_id="pytest-run",
        output_root=tmp_path / "output",
        audit_root=tmp_path / "audit",
        trace_path=tmp_path / "traces" / "trace.jsonl",
    )
    yield instance
    instance.close()


def test_pipeline_matches_labelled_samples(pipeline, sample_threads, sample_answers):
    subset = sample_threads[sample_threads.thread_id.isin(["T00001", "T00005", "T00006", "T00010", "T00028"])]
    results = pipeline.run_dataframe(subset)
    labels = pipeline.labels_from_results(results)
    expected = sample_answers.set_index("thread_id")["label"].to_dict()
    for thread_id, label in labels.items():
        assert label == expected[thread_id]
    assert len(results) == len(subset)
    assert pipeline.metrics.summary().verification_counts == {"PASSED": len(subset)}


def test_blacklisted_early_route_skips_retrieval(pipeline, sample_threads):
    subset = sample_threads[sample_threads.thread_id.isin(["T00006"])]
    result = pipeline.run_dataframe(subset)[0]
    trace = result.get("stage_trace", ())
    assert "understand_thread" in trace
    assert "build_evidence" in trace
    assert "construct_query" not in trace
    assert "hybrid_retrieve" not in trace


def test_price_case_uses_retrieval(pipeline, sample_threads):
    subset = sample_threads[sample_threads.thread_id.isin(["T00001"])]
    result = pipeline.run_dataframe(subset)[0]
    trace = result.get("stage_trace", ())
    assert "construct_query" in trace
    assert "hybrid_retrieve" in trace
    assert result["verification"].passed


def test_audit_persistence_one_record_per_thread(pipeline, sample_threads, tmp_path):
    subset = sample_threads[sample_threads.thread_id.isin(["T00001", "T00006"])]
    pipeline.run_dataframe(subset)
    audit_dir = tmp_path / "audit" / "pytest-run"
    lines = (audit_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {record["thread_id"] for record in records} == {"T00001", "T00006"}
    assert all(record["verification"]["status"] == "PASSED" for record in records)


def test_submission_writer_and_validator(tmp_path):
    template = tmp_path / "template.csv"
    pd.DataFrame({"thread_id": ["A", "B"], "label": ["ESCALATE:MISSING_INFO:"] * 2}).to_csv(template, index=False)
    output = tmp_path / "submission.csv"
    write_submission(
        {"A": "ACCEPT:ACCEPTABLE_PRICE:", "B": "COUNTER:COUNTER_OFFER:123"},
        template_path=template,
        output_path=output,
    )
    report = validate_submission_file(output, template_path=template)
    assert report["rows"] == 2
    assert report["valid_labels"]


def test_submission_missing_thread_rejected(tmp_path):
    template = tmp_path / "template.csv"
    pd.DataFrame({"thread_id": ["A", "B"], "label": ["ESCALATE:MISSING_INFO:"] * 2}).to_csv(template, index=False)
    with pytest.raises(SubmissionValidationError):
        write_submission({"A": "ACCEPT:ACCEPTABLE_PRICE:"}, template_path=template, output_path=tmp_path / "out.csv")


def test_submission_invalid_label_rejected(tmp_path):
    template = tmp_path / "template.csv"
    pd.DataFrame({"thread_id": ["A"], "label": ["ESCALATE:MISSING_INFO:"]}).to_csv(template, index=False)
    output = tmp_path / "submission.csv"
    pd.DataFrame({"thread_id": ["A"], "label": ["ACCEPT"]}).to_csv(output, index=False)
    with pytest.raises(SubmissionValidationError):
        validate_submission_file(output, template_path=template)
