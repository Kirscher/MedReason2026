from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from medreason_docker.parsing import normalize_mcq_answer, parse_reasoning_and_answer
from medreason_docker.retrieval import RetrievalBank
from medreason_docker.schema import MedReasonCase, MedReasonPrediction
from medreason_docker.systems.base import MedReasonSystem

SYSTEM_PROMPT = (
    "You are a medical visual reasoning assistant for the MedReason Challenge. "
    "Use the image evidence, the question, and the official options when present. "
    "Avoid unsupported visual claims. Return only valid JSON with keys "
    "`reasoning_trace` and `answer`."
)


class StrongBaselineSystem(MedReasonSystem):
    """Retrieval-augmented Qwen2.5-VL baseline with deterministic fallbacks."""

    def setup(self) -> None:
        self.retrieval_bank: RetrievalBank | None = None
        self.processor = None
        self.model = None
        self.torch = None
        self.mcq_adapter_name: str | None = None
        self.open_adapter_name: str | None = None

        if self.config.retrieval_bank_path and self.config.retrieval_bank_path.exists():
            print(f"[MedReason] loading retrieval bank: {self.config.retrieval_bank_path}", flush=True)
            self.retrieval_bank = RetrievalBank.from_json(self.config.retrieval_bank_path)
        else:
            if self.config.strict_resources:
                raise RuntimeError(
                    f"Strict resources enabled and retrieval bank is missing: "
                    f"{self.config.retrieval_bank_path}"
                )
            print("[MedReason] no retrieval bank found; retrieval few-shot disabled", flush=True)

        if self.config.disable_vlm:
            print("[MedReason] MEDREASON_DISABLE_VLM=1; using retrieval fallback only", flush=True)
            return
        if not self.config.model_path or not Path(self.config.model_path).exists():
            if self.config.strict_resources:
                raise RuntimeError(
                    f"Strict resources enabled and model checkpoint is missing: {self.config.model_path}"
                )
            print("[MedReason] model checkpoint not found; using retrieval fallback only", flush=True)
            return

        self._load_qwen(Path(self.config.model_path))

    def _load_qwen(self, model_path: Path) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Qwen mode requires torch, transformers, and qwen-vl-utils.") from exc

        quantization_config = None
        if self.config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("4-bit mode requires bitsandbytes support.") from exc

        dtype = "auto"
        if self.config.dtype not in {"", "auto"}:
            dtype = getattr(torch, self.config.dtype)

        load_errors: list[str] = []
        model = None
        for class_name in (
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ):
            try:
                transformers_mod = __import__("transformers", fromlist=[class_name])
                model_cls = getattr(transformers_mod, class_name)
                kwargs: dict[str, Any] = {
                    "torch_dtype": dtype,
                    "device_map": "auto",
                    "trust_remote_code": True,
                }
                if quantization_config is not None:
                    kwargs["quantization_config"] = quantization_config
                model = model_cls.from_pretrained(str(model_path), **kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                load_errors.append(f"{class_name}: {exc}")
        if model is None:
            raise RuntimeError("Unable to load Qwen-compatible VLM:\n" + "\n".join(load_errors))

        if self.config.lora_path:
            lora_path = Path(self.config.lora_path)
            if not lora_path.exists():
                raise RuntimeError(f"LoRA adapter path not found: {lora_path}")
            try:
                try:
                    import peft.tuners.lora.awq as peft_lora_awq  # type: ignore

                    peft_lora_awq.is_gptqmodel_available = lambda: False
                except Exception:
                    pass
                from peft import PeftModel  # type: ignore

                model = PeftModel.from_pretrained(model, str(lora_path), adapter_name="mcq")
                model.set_adapter("mcq")
                self.mcq_adapter_name = "mcq"
                print(f"[MedReason] loaded LoRA adapter from {lora_path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Unable to load LoRA adapter: {lora_path}") from exc
        if self.config.open_lora_path:
            if self.config.open_lora_path.strip().lower() in {"base", "none", "disabled"}:
                self.open_adapter_name = "base"
                print("[MedReason] open cases will run with LoRA adapters disabled", flush=True)
            else:
                self._load_open_lora_adapter(model)

        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = model.eval()
        if dtype is torch.float16:
            try:
                self.model.to(dtype=torch.float16)
            except Exception as exc:  # noqa: BLE001
                print(f"[MedReason] warning: unable to cast model to float16: {exc}", flush=True)
        if self.config.force_model_to_cuda and torch.cuda.is_available():
            self.model.to("cuda")
        self.torch = torch
        print(f"[MedReason] loaded VLM from {model_path}", flush=True)

    def _load_open_lora_adapter(self, model: Any) -> None:
        if self.config.open_lora_path:
            if not self.config.lora_path:
                raise RuntimeError("MEDREASON_OPEN_LORA_PATH requires MEDREASON_LORA_PATH to be set")
            open_lora_path = Path(self.config.open_lora_path)
            if not open_lora_path.exists():
                raise RuntimeError(f"Open LoRA adapter path not found: {open_lora_path}")
            try:
                model.load_adapter(str(open_lora_path), adapter_name="open")
                self.open_adapter_name = "open"
                print(f"[MedReason] loaded open LoRA adapter from {open_lora_path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Unable to load open LoRA adapter: {open_lora_path}") from exc

    def predict_case(self, case: MedReasonCase) -> MedReasonPrediction:
        if self.model is None or self.processor is None:
            return self._fallback_prediction(case)

        try:
            raw_text = self._generate(case)
        except Exception as exc:  # noqa: BLE001 - keep challenge run alive per case.
            self._clear_cuda_cache()
            return self._case_error_fallback(case, exc)
        reasoning_trace, answer = parse_reasoning_and_answer(raw_text, case)
        metadata: dict[str, Any] = {
            "system": "strong_baseline_qwen",
            "raw_response": self._trim(raw_text, 1200),
        }
        if case.task_type == "mcq":
            raw_vlm_answer = answer
            answer = normalize_mcq_answer(answer, case)
            vlm_answer = answer
            metadata["raw_vlm_answer"] = raw_vlm_answer
            metadata["vlm_answer"] = vlm_answer
            if self.mcq_adapter_name:
                metadata["active_lora_adapter"] = self.mcq_adapter_name
            retrieval_answer, retrieval_confidence, retrieval_meta = self._option_aware_mcq_answer(case)
            if retrieval_answer and self.config.mcq_policy in {"option_aware", "hybrid_low_confidence"}:
                metadata["mcq_policy"] = self.config.mcq_policy
                metadata["option_score_mode"] = self.config.option_score_mode
                metadata["retrieval_answer"] = retrieval_answer
                metadata["retrieval_confidence"] = round(retrieval_confidence, 4)
                metadata["retrieval_option_scores"] = retrieval_meta
                answer = retrieval_answer
                if (
                    self.config.mcq_policy == "hybrid_low_confidence"
                    and vlm_answer in case.option_labels
                    and retrieval_confidence < self.config.vlm_override_confidence_threshold
                ):
                    answer = vlm_answer
                    metadata["mcq_override"] = "vlm_low_retrieval_confidence"
                    metadata["vlm_override_confidence_threshold"] = (
                        self.config.vlm_override_confidence_threshold
                    )
        elif self.retrieval_bank:
            if self.open_adapter_name:
                metadata["active_lora_adapter"] = self.open_adapter_name
            open_candidates = self._open_retrieval_candidates(case, top_k=3)
            metadata["open_retrieval_candidates"] = open_candidates
            answer = self._apply_open_answer_policy(answer, open_candidates, metadata)
        if not answer.strip() or self._looks_placeholder(answer):
            answer = self._fallback_answer(case)
        if not reasoning_trace.strip() or self._looks_placeholder(reasoning_trace):
            reasoning_trace = self._fallback_reasoning(case)

        return MedReasonPrediction(
            case_id=case.case_id,
            task_type=case.task_type,
            answer=answer,
            reasoning_trace=self._trim(reasoning_trace, 1200),
            confidence=None,
            metadata=metadata,
        )

    def _case_error_fallback(self, case: MedReasonCase, exc: Exception) -> MedReasonPrediction:
        fallback = self._fallback_prediction(case)
        metadata = dict(fallback.metadata)
        metadata.update(
            {
                "system": "strong_baseline_case_error_fallback",
                "error_type": type(exc).__name__,
                "error_message": self._trim(str(exc), 500),
            }
        )
        return MedReasonPrediction(
            case_id=fallback.case_id,
            task_type=fallback.task_type,
            answer=fallback.answer,
            reasoning_trace=fallback.reasoning_trace,
            confidence=0.0,
            metadata=metadata,
        )

    def _clear_cuda_cache(self) -> None:
        torch_mod = getattr(self, "torch", None)
        cuda = getattr(torch_mod, "cuda", None)
        if cuda is None:
            return
        try:
            if cuda.is_available():
                cuda.empty_cache()
        except Exception:
            return

    def _generate(self, case: MedReasonCase) -> str:
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Qwen generation requires qwen-vl-utils.") from exc

        messages = self._messages(case)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        target_dtype = None
        if self.config.dtype not in {"", "auto"}:
            target_dtype = getattr(self.torch, self.config.dtype)
            for key, value in list(inputs.items()):
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    inputs[key] = value.to(dtype=target_dtype)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            generation_kwargs.update({"temperature": self.config.temperature, "top_p": self.config.top_p})

        active_adapter = self._adapter_for_case(case)
        adapter_context = None
        if active_adapter == "base" and hasattr(self.model, "disable_adapter"):
            adapter_context = self.model.disable_adapter()
        elif active_adapter and hasattr(self.model, "set_adapter"):
            self.model.set_adapter(active_adapter)

        if adapter_context is None:
            with self.torch.inference_mode():
                generated = self._generate_tokens(inputs, generation_kwargs, target_dtype)
        else:
            with adapter_context:
                with self.torch.inference_mode():
                    generated = self._generate_tokens(inputs, generation_kwargs, target_dtype)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0].strip()

    def _generate_tokens(
        self,
        inputs: Any,
        generation_kwargs: dict[str, Any],
        target_dtype: Any,
    ) -> Any:
        if (
            target_dtype is self.torch.float16
            and self.torch.cuda.is_available()
            and str(self.model.device).startswith("cuda")
        ):
            with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                return self.model.generate(**inputs, **generation_kwargs)
        return self.model.generate(**inputs, **generation_kwargs)

    def _adapter_for_case(self, case: MedReasonCase) -> str | None:
        if case.task_type == "open" and self.open_adapter_name:
            return self.open_adapter_name
        if case.task_type == "mcq" and self.mcq_adapter_name:
            return self.mcq_adapter_name
        return self.mcq_adapter_name

    def _messages(self, case: MedReasonCase) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for path in case.image_paths:
            content.append({"type": "image", "image": str(path)})
        content.append({"type": "text", "text": self._prompt(case)})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _prompt(self, case: MedReasonCase) -> str:
        examples = self._few_shot_examples(case)
        example_block = "\n\n".join(examples)
        if case.task_type == "mcq":
            options = "\n".join(f"{opt.label}. {opt.text}" for opt in case.options)
            return (
                "Task: closed-ended medical VQA.\n"
                "Select exactly one official option label. Do not answer with option text.\n\n"
                f"{example_block}\n\n"
                f"Question:\n{case.question}\n\n"
                f"Options:\n{options}\n\n"
                "Return only a JSON object. The values must be specific to this case. "
                "Use this schema: {\"reasoning_trace\": string, \"answer\": one_option_label}"
            )
        if self.config.open_prompt_style == "evidence_first":
            return (
                "Task: open-ended medical visual reasoning.\n"
                "Use the image first. Retrieved examples are analogies only; do not copy a retrieved "
                "answer unless the same visual finding is present in this case.\n"
                "Return a concise image-grounded reasoning trace and a concise final answer. "
                "Prefer the directly visible abnormality, temporal change, morphology, location, "
                "or spatial relationship requested by the question. Do not hallucinate findings "
                "that are not visible.\n\n"
                f"{self._metadata_block(case)}\n\n"
                f"{example_block}\n\n"
                f"Question:\n{case.question}\n\n"
                "Return only a JSON object. The values must be specific to this case and must not copy "
                "the schema text. Use this schema: {\"reasoning_trace\": string, \"answer\": string}"
            )
        if self.config.open_prompt_style == "modality_guard":
            meta_block = self._metadata_block(case)
            return (
                "Task: open-ended medical visual reasoning.\n"
                + (f"{meta_block}\n\n" if meta_block else "")
                + "Ground your answer exclusively in this case's modality and anatomical region. "
                "Do not describe findings, structures, or pathologies from a different modality or body region.\n"
                "Retrieved examples are analogies only; if a retrieved answer describes a different modality "
                "or organ than indicated above, discard it.\n\n"
                f"{example_block}\n\n"
                f"Question:\n{case.question}\n\n"
                "Return only a JSON object. The values must be specific to this case and must not copy "
                "the schema text. Use this schema: {\"reasoning_trace\": string, \"answer\": string}"
            )
        return (
            "Task: open-ended medical visual reasoning.\n"
            "Return a concise image-grounded reasoning trace and a concise final answer. "
            "Do not hallucinate findings that are not visible.\n\n"
            f"{example_block}\n\n"
            f"Question:\n{case.question}\n\n"
            "Return only a JSON object. The values must be specific to this case and must not copy "
            "the schema text. Use this schema: {\"reasoning_trace\": string, \"answer\": string}"
        )

    def _few_shot_examples(self, case: MedReasonCase) -> list[str]:
        top_k = self.config.open_top_k_examples if case.task_type == "open" else self.config.top_k_examples
        if not self.retrieval_bank or top_k <= 0:
            return []
        query = " ".join([case.question, " ".join(opt.text for opt in case.options)])
        hits = self.retrieval_bank.search(query, task_type=case.task_type, top_k=top_k)
        blocks: list[str] = []
        for idx, hit in enumerate(hits, start=1):
            ex = hit.example
            answer = str(ex.get("answer", "")).strip()
            if not answer:
                continue
            visual = str(ex.get("visual_description", "")).strip()
            blocks.append(
                f"Retrieved example {idx}:\n"
                f"Question: {self._trim(str(ex.get('question', '')), 450)}\n"
                f"Known answer: {self._trim(answer, 180)}\n"
                f"Visual cue: {self._trim(visual, 450)}"
            )
        return blocks

    def _metadata_block(self, case: MedReasonCase) -> str:
        if not case.metadata:
            return ""
        ordered = []
        for key in ("modality", "organ_system", "task", "subtype"):
            value = case.metadata.get(key)
            if value:
                ordered.append(f"{key}: {value}")
        return "Case context: " + "; ".join(ordered) if ordered else ""

    def _open_retrieval_candidates(self, case: MedReasonCase, top_k: int) -> list[dict[str, Any]]:
        if not self.retrieval_bank:
            return []
        metadata_text = " ".join(str(value) for value in case.metadata.values() if value)
        query = " ".join([case.question, metadata_text])
        candidates = []
        for rank, hit in enumerate(self.retrieval_bank.search(query, task_type="open", top_k=top_k), start=1):
            candidates.append(
                {
                    "rank": rank,
                    "case_id": hit.example.get("case_id"),
                    "score": round(hit.score, 4),
                    "answer": self._trim(str(hit.example.get("answer", "")), 180),
                }
            )
        return candidates

    def _apply_open_answer_policy(
        self,
        answer: str,
        candidates: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> str:
        if self.config.open_answer_policy != "retrieval_gap" or not candidates:
            return answer
        top = candidates[0]
        top_answer = str(top.get("answer") or "").strip()
        top_score = float(top.get("score") or 0.0)
        second_score = float(candidates[1].get("score") or 0.0) if len(candidates) > 1 else 0.0
        gap = top_score - second_score
        if (
            top_answer
            and top_score >= self.config.open_retrieval_min_top_score
            and gap >= self.config.open_retrieval_min_score_gap
        ):
            metadata["open_override"] = "retrieval_score_gap"
            metadata["open_override_previous_answer"] = answer
            metadata["open_override_score_gap"] = round(gap, 4)
            metadata["open_answer_policy"] = self.config.open_answer_policy
            return top_answer
        return answer

    def _fallback_prediction(self, case: MedReasonCase) -> MedReasonPrediction:
        answer = self._fallback_answer(case)
        if case.task_type == "mcq":
            reasoning_trace = self._fallback_reasoning(case)
        else:
            reasoning_trace = self._fallback_reasoning(case)
        return MedReasonPrediction(
            case_id=case.case_id,
            task_type=case.task_type,
            answer=answer,
            reasoning_trace=reasoning_trace,
            confidence=0.0,
            metadata={"system": "strong_baseline_retrieval_fallback"},
        )

    def _fallback_answer(self, case: MedReasonCase) -> str:
        if self.retrieval_bank:
            if case.task_type == "mcq":
                answer, _, _ = self._option_aware_mcq_answer(case)
                if answer:
                    return answer
            query = " ".join([case.question, " ".join(opt.text for opt in case.options)])
            hits = self.retrieval_bank.search(query, task_type=case.task_type, top_k=5)
            if case.task_type == "open" and hits:
                answer = str(hits[0].example.get("answer", "")).strip()
                if answer:
                    return answer
        if case.task_type == "mcq":
            return case.options[0].label
        return "Unable to determine confidently from the available visual evidence."

    def _fallback_reasoning(self, case: MedReasonCase) -> str:
        if self.retrieval_bank:
            query = " ".join([case.question, " ".join(opt.text for opt in case.options)])
            hits = self.retrieval_bank.search(query, task_type=case.task_type, top_k=1)
            if hits:
                visual = str(hits[0].example.get("visual_description", "")).strip()
                if visual:
                    return (
                        "Fallback retrieval found a similar training case. "
                        f"Retrieved visual cue: {self._trim(visual, 600)}"
                    )
        if case.task_type == "mcq":
            answer, confidence, _ = self._option_aware_mcq_answer(case)
            if answer:
                return (
                    "Fallback compared each official option with training examples using known "
                    f"correct-answer text and selected option {answer} "
                    f"(retrieval confidence {confidence:.2f})."
                )
            return "Fallback selected the first official option because no retrieval prior was available."
        return "Fallback retrieval was used because the model output was empty or non-specific."

    def _option_aware_mcq_answer(self, case: MedReasonCase) -> tuple[str | None, float, dict[str, float]]:
        if not self.retrieval_bank or case.task_type != "mcq" or not case.options:
            return None, 0.0, {}

        metadata_text = " ".join(str(value) for value in case.metadata.values() if value)
        scores: dict[str, float] = {}
        for option in case.options:
            query = " ".join([case.question, option.text, metadata_text])
            hits = self.retrieval_bank.search_correct_answer_examples(
                query, top_k=self.config.option_retrieval_top_k
            )
            if not hits:
                scores[option.label] = 0.0
                continue
            hit_scores = [max(hit.score, 0.0) for hit in hits]
            scores[option.label] = self._score_option_hits(hit_scores)

        if not scores or max(scores.values()) <= 0:
            return None, 0.0, scores
        answer = max(scores, key=scores.get)
        total = sum(max(score, 0.0) for score in scores.values()) or 1.0
        confidence = max(scores.values()) / total
        return answer, confidence, scores

    def _score_option_hits(self, scores: list[float]) -> float:
        if not scores:
            return 0.0
        if self.config.option_score_mode == "top1":
            return scores[0]
        if self.config.option_score_mode == "sum3":
            return sum(scores[:3])
        if self.config.option_score_mode == "top1_plus_sum5":
            return scores[0] + sum(scores[:5])
        if self.config.option_score_mode == "ranked5":
            return sum(score / (rank + 1) for rank, score in enumerate(scores[:5]))
        return sum(scores[:5])

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _looks_placeholder(text: str) -> bool:
        normalized = re.sub(r"[\s_`\"'.:-]+", " ", text.strip().lower()).strip()
        return normalized in {
            "string",
            "visual evidence",
            "final answer",
            "concise final answer",
            "brief image grounded rationale",
            "image grounded evidence and reasoning",
        }
