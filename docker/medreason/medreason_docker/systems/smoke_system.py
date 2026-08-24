from __future__ import annotations

from medreason_docker.schema import MedReasonCase, MedReasonPrediction
from medreason_docker.systems.base import MedReasonSystem


class SmokeMedReasonSystem(MedReasonSystem):
    """Lightweight deterministic system used only for I/O smoke tests.

    This is not a competitive baseline. It verifies that Docker mounting,
    input parsing, output writing, and validation are functioning.
    """

    def predict_case(self, case: MedReasonCase) -> MedReasonPrediction:
        if case.task_type == "mcq":
            answer = case.options[0].label
            return MedReasonPrediction(
                case_id=case.case_id,
                task_type=case.task_type,
                reasoning_trace="Smoke-test system selected the first available option.",
                answer=answer,
                confidence=0.0,
                metadata={"system": "smoke"},
            )

        return MedReasonPrediction(
            case_id=case.case_id,
            task_type=case.task_type,
            reasoning_trace=(
                "Smoke-test system does not inspect image evidence. "
                "This placeholder trace is provided only to validate the required output schema."
            ),
            answer="Unable to determine from the smoke-test system.",
            confidence=0.0,
            metadata={"system": "smoke"},
        )
