from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration for the participant container.

    The official evaluator will mount `/input` and `/output`. Environment variables
    are provided mainly to make local testing easier.
    """

    input_dir: Path
    output_dir: Path
    output_file: Path
    system_name: str
    submission_name: str
    model_path: str | None
    lora_path: str | None
    open_lora_path: str | None
    retrieval_bank_path: Path | None
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k_examples: int
    open_top_k_examples: int
    open_prompt_style: str
    open_answer_policy: str
    open_retrieval_min_score_gap: float
    open_retrieval_min_top_score: float
    mcq_policy: str
    option_retrieval_top_k: int
    option_score_mode: str
    vlm_override_confidence_threshold: float
    disable_vlm: bool
    load_in_4bit: bool
    force_model_to_cuda: bool
    strict_resources: bool
    dtype: str
    log_every: int

    @property
    def cases_file(self) -> Path:
        return self.input_dir / "cases.json"

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        input_dir = Path(os.environ.get("MEDREASON_INPUT_DIR", "/input"))
        output_dir = Path(os.environ.get("MEDREASON_OUTPUT_DIR", "/output"))
        output_file = Path(os.environ.get("MEDREASON_OUTPUT_FILE", str(output_dir / "results.json")))
        return cls(
            input_dir=input_dir,
            output_dir=output_dir,
            output_file=output_file,
            system_name=os.environ.get("MEDREASON_SYSTEM", "custom").strip().lower(),
            submission_name=os.environ.get("MEDREASON_SUBMISSION_NAME", "MedReason system submission"),
            model_path=os.environ.get("MEDREASON_MODEL_PATH"),
            lora_path=os.environ.get("MEDREASON_LORA_PATH"),
            open_lora_path=os.environ.get("MEDREASON_OPEN_LORA_PATH"),
            retrieval_bank_path=Path(os.environ["MEDREASON_RETRIEVAL_BANK"])
            if os.environ.get("MEDREASON_RETRIEVAL_BANK")
            else None,
            max_new_tokens=int(os.environ.get("MEDREASON_MAX_NEW_TOKENS", "512")),
            temperature=float(os.environ.get("MEDREASON_TEMPERATURE", "0.0")),
            top_p=float(os.environ.get("MEDREASON_TOP_P", "1.0")),
            top_k_examples=int(os.environ.get("MEDREASON_TOP_K_EXAMPLES", "3")),
            open_top_k_examples=int(
                os.environ.get(
                    "MEDREASON_OPEN_TOP_K_EXAMPLES",
                    os.environ.get("MEDREASON_TOP_K_EXAMPLES", "3"),
                )
            ),
            open_prompt_style=os.environ.get("MEDREASON_OPEN_PROMPT_STYLE", "baseline").strip().lower(),
            open_answer_policy=os.environ.get("MEDREASON_OPEN_ANSWER_POLICY", "vlm").strip().lower(),
            open_retrieval_min_score_gap=float(
                os.environ.get("MEDREASON_OPEN_RETRIEVAL_MIN_SCORE_GAP", "0.04")
            ),
            open_retrieval_min_top_score=float(
                os.environ.get("MEDREASON_OPEN_RETRIEVAL_MIN_TOP_SCORE", "0.0")
            ),
            mcq_policy=os.environ.get("MEDREASON_MCQ_POLICY", "option_aware").strip().lower(),
            option_retrieval_top_k=int(os.environ.get("MEDREASON_OPTION_RETRIEVAL_TOP_K", "5")),
            option_score_mode=os.environ.get("MEDREASON_OPTION_SCORE_MODE", "sum5").strip().lower(),
            vlm_override_confidence_threshold=float(
                os.environ.get("MEDREASON_VLM_OVERRIDE_CONFIDENCE_THRESHOLD", "0.257")
            ),
            disable_vlm=os.environ.get("MEDREASON_DISABLE_VLM", "0").strip().lower()
            in {"1", "true", "yes"},
            load_in_4bit=os.environ.get("MEDREASON_LOAD_IN_4BIT", "0").strip().lower()
            in {"1", "true", "yes"},
            force_model_to_cuda=os.environ.get("MEDREASON_FORCE_MODEL_TO_CUDA", "0").strip().lower()
            in {"1", "true", "yes"},
            strict_resources=os.environ.get("MEDREASON_STRICT_RESOURCES", "0").strip().lower()
            in {"1", "true", "yes"},
            dtype=os.environ.get("MEDREASON_DTYPE", "auto").strip().lower(),
            log_every=max(1, int(os.environ.get("MEDREASON_LOG_EVERY", "1"))),
        )
