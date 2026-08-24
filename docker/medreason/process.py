#!/usr/bin/env python3
"""MedReason participant-container entry point.

The official evaluator will run the submitted container with a mounted input directory
and output directory. The container must read `/input/cases.json` and write
`/output/results.json`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from medreason_docker.config import RuntimeConfig
from medreason_docker.io import load_cases, write_predictions
from medreason_docker.runner import run_system
from medreason_docker.validation import validate_predictions_against_cases


def main() -> int:
    config = RuntimeConfig.from_env()
    config.output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[MedReason] input_dir={config.input_dir}", flush=True)
    print(f"[MedReason] output_file={config.output_file}", flush=True)
    print(f"[MedReason] system={config.system_name}", flush=True)

    cases = load_cases(config.cases_file, input_dir=config.input_dir)
    print(f"[MedReason] loaded {len(cases)} cases", flush=True)

    predictions = run_system(cases=cases, config=config)
    validate_predictions_against_cases(predictions, cases)
    write_predictions(predictions, config.output_file, submission_name=config.submission_name)

    print(f"[MedReason] wrote predictions to {config.output_file}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - challenge entry point should fail loudly.
        print(f"[MedReason][ERROR] {exc}", file=sys.stderr, flush=True)
        raise
