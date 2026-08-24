from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker" / "medreason"))
sys.path.insert(0, str(ROOT))

from medreason_docker.config import RuntimeConfig
from medreason_docker.parsing import normalize_mcq_answer, parse_reasoning_and_answer
from medreason_docker.schema import MedReasonCase, MedReasonOption
from medreason_docker.systems.strong_baseline_system import StrongBaselineSystem
from finetune.build_sft_dataset import permute_mcq_options, prompt_for_case, response_for_case
from scripts import validate_predictions


def mcq_case() -> MedReasonCase:
    return MedReasonCase(
        case_id="case_mcq",
        task_type="mcq",
        question="Which finding is present?",
        image_paths=(Path("image.png"),),
        options=(
            MedReasonOption(label="A", text="Normal study"),
            MedReasonOption(label="B", text="Pneumothorax"),
            MedReasonOption(label="C", text="Pleural effusion"),
        ),
    )


class ParsingTests(unittest.TestCase):
    def test_mcq_normalization_does_not_treat_article_a_as_label(self) -> None:
        case = mcq_case()
        self.assertEqual(
            normalize_mcq_answer("There is a pneumothorax.", case),
            "There is a pneumothorax.",
        )
        self.assertEqual(normalize_mcq_answer("A pneumothorax is present.", case), "A pneumothorax is present.")
        self.assertEqual(normalize_mcq_answer("This is a CT image.", case), "This is a CT image.")

    def test_mcq_normalization_accepts_explicit_label_forms(self) -> None:
        case = mcq_case()
        for answer in ["B", "(B)", "B.", "B. Pneumothorax", "Option B", "Answer: B"]:
            self.assertEqual(normalize_mcq_answer(answer, case), "B")

    def test_mcq_normalization_maps_exact_option_text(self) -> None:
        self.assertEqual(normalize_mcq_answer("Pneumothorax", mcq_case()), "B")

    def test_json_parser_uses_first_valid_object_with_surrounding_noise(self) -> None:
        case = MedReasonCase(
            case_id="case_open",
            task_type="open",
            question="Describe the image.",
            image_paths=(Path("image.png"),),
        )
        text = 'prefix {"reasoning_trace": "visible cue", "answer": "final"} suffix {bad}'
        self.assertEqual(parse_reasoning_and_answer(text, case), ("visible cue", "final"))


class SftDatasetTests(unittest.TestCase):
    def test_mcq_option_permutation_preserves_correct_option_text(self) -> None:
        case = {
            "case_id": "case_1",
            "question type": "mcq",
            "question": "Which finding is present?",
            "answer": "B",
            "A": "A) Normal study",
            "B": "B) Pneumothorax",
            "C": "C) Pleural effusion",
        }
        permuted = permute_mcq_options(case, seed=81, variant=1)

        self.assertNotEqual(
            [permuted[label] for label in "ABC"],
            [case[label] for label in "ABC"],
        )
        self.assertIn("Pneumothorax", permuted[permuted["answer"]])

    def test_mcq_answer_first_response_puts_label_before_reasoning(self) -> None:
        case = {
            "question type": "mcq",
            "answer": "C",
            "C": "C) Pleural effusion",
        }
        response = response_for_case(case, response_style="mcq_option_answer_first")

        self.assertLess(response.index('"answer"'), response.index('"reasoning_trace"'))
        self.assertEqual(json.loads(response)["answer"], "C")
        self.assertEqual(json.loads(response)["reasoning_trace"], "Pleural effusion")

    def test_runtime_mcq_prompt_includes_retrieval_before_current_question(self) -> None:
        case = {
            "question type": "mcq",
            "question": "Current question",
            "answer": "A",
            "A": "A) First",
            "B": "B) Second",
        }
        prompt = prompt_for_case(
            case,
            [{"question": "Example question", "answer": "B", "visual_description": "Example cue"}],
            prompt_style="runtime_mcq",
        )

        self.assertLess(prompt.index("Retrieved example 1"), prompt.index("Current question"))
        self.assertIn("A. First", prompt)
        self.assertIn("B. Second", prompt)


class StrongBaselineFallbackTests(unittest.TestCase):
    def test_generation_exception_returns_valid_fallback_prediction(self) -> None:
        config = RuntimeConfig(
            input_dir=Path("/input"),
            output_dir=Path("/output"),
            output_file=Path("/output/results.json"),
            system_name="strong_baseline",
            submission_name="test",
            model_path=None,
            lora_path=None,
            open_lora_path=None,
            retrieval_bank_path=None,
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
            top_k_examples=0,
            open_top_k_examples=0,
            open_prompt_style="baseline",
            open_answer_policy="vlm",
            open_retrieval_min_score_gap=0.04,
            open_retrieval_min_top_score=0.0,
            mcq_policy="hybrid_low_confidence",
            option_retrieval_top_k=5,
            option_score_mode="sum5",
            vlm_override_confidence_threshold=0.239,
            disable_vlm=False,
            load_in_4bit=False,
            force_model_to_cuda=False,
            strict_resources=False,
            dtype="auto",
            log_every=1,
        )
        system = StrongBaselineSystem(config=config)
        system.retrieval_bank = None
        system.model = object()
        system.processor = object()
        system.torch = None

        with patch.object(system, "_generate", side_effect=RuntimeError("synthetic failure")):
            prediction = system.predict_case(mcq_case())

        self.assertEqual(prediction.answer, "A")
        self.assertEqual(prediction.metadata["system"], "strong_baseline_case_error_fallback")
        self.assertEqual(prediction.metadata["error_type"], "RuntimeError")


class ValidationScriptTests(unittest.TestCase):
    def test_task_type_mismatch_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases_path = tmp_path / "cases.json"
            results_path = tmp_path / "results.json"
            cases_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "case_1",
                                "task_type": "mcq",
                                "image_path": "image.png",
                                "question": "Question?",
                                "options": [{"label": "A", "text": "Option A"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            results_path.write_text(
                json.dumps(
                    {
                        "answers": [
                            {
                                "case_id": "case_1",
                                "task_type": "open",
                                "answer": "A",
                                "reasoning_trace": "trace",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "validate_predictions.py",
                "--cases-json",
                str(cases_path),
                "--results-json",
                str(results_path),
            ]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                validate_predictions.main()


if __name__ == "__main__":
    unittest.main()
