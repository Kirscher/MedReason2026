from __future__ import annotations

from .schema import MedReasonCase, MedReasonPrediction


def validate_predictions_against_cases(
    predictions: list[MedReasonPrediction],
    cases: list[MedReasonCase],
) -> None:
    if len(predictions) != len(cases):
        raise ValueError(f"Expected {len(cases)} predictions but received {len(predictions)}")

    cases_by_id = {case.case_id: case for case in cases}
    seen: set[str] = set()
    for pred in predictions:
        if pred.case_id in seen:
            raise ValueError(f"Duplicate prediction for case_id={pred.case_id}")
        seen.add(pred.case_id)
        if pred.case_id not in cases_by_id:
            raise ValueError(f"Prediction contains unknown case_id={pred.case_id}")

        case = cases_by_id[pred.case_id]
        if pred.task_type != case.task_type:
            raise ValueError(
                f"case_id={pred.case_id} task_type mismatch: expected {case.task_type}, got {pred.task_type}"
            )
        if not isinstance(pred.answer, str) or not pred.answer.strip():
            raise ValueError(f"case_id={pred.case_id} answer must be a non-empty string")

        if case.task_type == "mcq":
            answer = pred.answer.strip()
            if answer not in case.option_labels:
                raise ValueError(
                    f"case_id={pred.case_id} MCQ answer must be one of {sorted(case.option_labels)}, got {answer!r}"
                )
        else:
            if not isinstance(pred.reasoning_trace, str) or not pred.reasoning_trace.strip():
                raise ValueError(
                    f"case_id={pred.case_id} open-ended cases require a non-empty reasoning_trace"
                )
