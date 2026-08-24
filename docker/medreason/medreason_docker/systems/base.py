from __future__ import annotations

from abc import ABC, abstractmethod

from medreason_docker.config import RuntimeConfig
from medreason_docker.schema import MedReasonCase, MedReasonPrediction


class MedReasonSystem(ABC):
    """Base interface for a complete MedReason MLLM system.

    A submission may use a single MLLM, multiple models, retrieval modules,
    agentic workflows, visual verification, self-consistency, rule-based
    post-processing, or uncertainty estimation. The only official requirement is
    that the container follows the input/output contract and runs fully inside
    the submitted Docker image during evaluation.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config

    def setup(self) -> None:
        """Load checkpoints, processors, indices, or any other resources."""

    def teardown(self) -> None:
        """Release resources if needed."""

    @abstractmethod
    def predict_case(self, case: MedReasonCase) -> MedReasonPrediction:
        """Run the full participant system on one case."""
