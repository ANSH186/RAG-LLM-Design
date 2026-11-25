"""Validate a submission CSV against the official template."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.submission import validate_submission_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--template", default="data/raw/sample_submission.csv")
    args = parser.parse_args()
    report = validate_submission_file(args.submission, template_path=args.template)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
