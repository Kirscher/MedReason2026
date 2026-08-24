from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TaskType = Literal["mcq", "open"]


@dataclass(frozen=True)
class MedReasonOption:
    label: str
    text: str


@dataclass(frozen=True)
class MedReasonCase:
    case_id: str
    task_type: TaskType
    question: str
    image_paths: tuple[Path, ...]
    options: tuple[MedReasonOption, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def option_labels(self) -> set[str]:
        return {option.label for option in self.options}

    @property
    def primary_image_path(self) -> Path | None:
        return self.image_paths[0] if self.image_paths else None


@dataclass(frozen=True)
class MedReasonPrediction:
    case_id: str
    task_type: TaskType
    answer: str
    reasoning_trace: str = ""
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
