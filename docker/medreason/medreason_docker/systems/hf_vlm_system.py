from __future__ import annotations

from typing import Any

from medreason_docker.image_utils import load_images
from medreason_docker.parsing import normalize_mcq_answer, parse_reasoning_and_answer
from medreason_docker.prompting import build_prompt
from medreason_docker.schema import MedReasonCase, MedReasonPrediction
from medreason_docker.systems.base import MedReasonSystem


class HuggingFaceVLMSystem(MedReasonSystem):
    """Example single-model HuggingFace VLM system.

    This file is provided as a reference implementation, not as a restriction on
    submissions. Participants may ignore it and implement any complete MLLM
    system in `custom_system.py` or in a new system class.
    """

    def setup(self) -> None:
        if not self.config.model_path:
            raise ValueError(
                "MEDREASON_MODEL_PATH must be set when MEDREASON_SYSTEM=hf_vlm. "
                "For smoke testing, use MEDREASON_SYSTEM=smoke."
            )
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "HuggingFace dependencies are not installed. Add torch/transformers/accelerate "
                "to requirements.txt or use requirements-extra-hf.txt as a starting point."
            ) from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.config.model_path, trust_remote_code=True)

        last_error: Exception | None = None
        for model_cls in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
            try:
                self.model = model_cls.from_pretrained(
                    self.config.model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        else:
            raise RuntimeError(f"Unable to load model from {self.config.model_path}") from last_error

        self.model.eval()

    def predict_case(self, case: MedReasonCase) -> MedReasonPrediction:
        images = load_images(case.image_paths)
        prompt = build_prompt(case)
        raw_text = self._generate(images=images, prompt=prompt)
        reasoning_trace, answer = parse_reasoning_and_answer(raw_text, case)
        if case.task_type == "mcq":
            answer = normalize_mcq_answer(answer, case)
        return MedReasonPrediction(
            case_id=case.case_id,
            task_type=case.task_type,
            reasoning_trace=reasoning_trace or raw_text,
            answer=answer,
            confidence=None,
            metadata={"system": "hf_vlm", "raw_response": raw_text[:2000]},
        )

    def _generate(self, images: list[Any], prompt: str) -> str:
        # Most modern HF VLM processors support a chat-template pathway. The code
        # below intentionally uses a conservative fallback so that participants can
        # adapt it to their exact model family.
        processor = self.processor
        model = self.model
        torch = self.torch

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "image": image} for image in images),
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt")
        except Exception:
            # Generic fallback for processors that accept `text` and `images`.
            image_arg: Any = images[0] if len(images) == 1 else images
            inputs = processor(text=prompt, images=image_arg, return_tensors="pt")

        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
            )

        # Decode only generated continuation when possible.
        input_len = inputs.get("input_ids").shape[-1] if "input_ids" in inputs else 0
        continuation = generated[:, input_len:] if input_len else generated
        decoded = processor.batch_decode(continuation, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""
