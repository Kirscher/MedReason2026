from __future__ import annotations

from .config import RuntimeConfig
from .schema import MedReasonCase, MedReasonPrediction
from .systems.base import MedReasonSystem
from .systems.custom_system import CustomMedReasonSystem
from .systems.hf_vlm_system import HuggingFaceVLMSystem
from .systems.smoke_system import SmokeMedReasonSystem
from .systems.strong_baseline_system import StrongBaselineSystem


def create_system(config: RuntimeConfig) -> MedReasonSystem:
    registry: dict[str, type[MedReasonSystem]] = {
        "custom": CustomMedReasonSystem,
        "smoke": SmokeMedReasonSystem,
        "hf_vlm": HuggingFaceVLMSystem,
        "huggingface": HuggingFaceVLMSystem,
        "strong_baseline": StrongBaselineSystem,
        "strong": StrongBaselineSystem,
    }
    try:
        system_cls = registry[config.system_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown MEDREASON_SYSTEM={config.system_name!r}. "
            f"Available systems: {sorted(registry)}"
        ) from exc
    return system_cls(config=config)


def run_system(cases: list[MedReasonCase], config: RuntimeConfig) -> list[MedReasonPrediction]:
    system = create_system(config)
    system.setup()
    predictions: list[MedReasonPrediction] = []
    for idx, case in enumerate(cases, start=1):
        if idx == 1 or idx == len(cases) or idx % config.log_every == 0:
            print(f"[MedReason] predicting {idx}/{len(cases)} case_id={case.case_id}", flush=True)
        pred = system.predict_case(case)
        predictions.append(pred)
    system.teardown()
    return predictions
