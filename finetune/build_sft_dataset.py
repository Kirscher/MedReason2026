from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


OPTION_LABELS = ("A", "B", "C", "D", "E")


def load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("cases") or payload.get("answers") or payload
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a list or cases/answers")
    return items


def task_type(case: dict[str, Any]) -> str:
    raw = str(case.get("task_type") or case.get("question type") or "").strip().lower()
    return "open" if raw in {"open", "open-ended", "open_ended"} else "mcq"


def option_text(case: dict[str, Any], label: str) -> str:
    text = str(case.get(label) or "").strip()
    prefix = f"{label})"
    return text[len(prefix) :].strip() if text.startswith(prefix) else text


def options_for_case(case: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"label": label, "text": option_text(case, label)}
        for label in OPTION_LABELS
        if str(case.get(label) or "").strip()
    ]


def permute_mcq_options(case: dict[str, Any], seed: int, variant: int) -> dict[str, Any]:
    labels = [option["label"] for option in options_for_case(case)]
    if len(labels) < 2:
        return dict(case)

    shuffled = list(labels)
    rng = random.Random(f"{seed}:{case.get('case_id', '')}:{variant}")
    while shuffled == labels:
        rng.shuffle(shuffled)

    correct_label = str(case.get("answer") or "").strip().upper()
    remapped = dict(case)
    remapped_answer = correct_label
    for new_label, old_label in zip(labels, shuffled):
        remapped[new_label] = f"{new_label}) {option_text(case, old_label)}"
        if old_label == correct_label:
            remapped_answer = new_label
    remapped["answer"] = remapped_answer
    return remapped


def excluded_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        for item in load_items(path):
            if "case_id" in item:
                ids.add(str(item["case_id"]))
    return ids


def image_path_for_case(case: dict[str, Any], train_img_dir: Path) -> str:
    raw = str(case.get("image_path") or "")
    name = Path(raw).name
    return str((train_img_dir / name).resolve())


def prompt_for_case(case: dict[str, Any], examples: list[dict[str, Any]], prompt_style: str = "baseline") -> str:
    task = task_type(case)
    if task == "mcq" and prompt_style == "runtime_mcq":
        example_blocks = []
        for idx, ex in enumerate(examples, start=1):
            example_blocks.append(
                "\n".join(
                    [
                        f"Retrieved example {idx}:",
                        f"Question: {str(ex.get('question', '')).strip()}",
                        f"Known answer: {str(ex.get('answer', '')).strip()}",
                        f"Visual cue: {str(ex.get('visual_description', '')).strip()}",
                    ]
                )
            )
        example_block = "\n\n".join(example_blocks)
        options = "\n".join(f"{opt['label']}. {opt['text']}" for opt in options_for_case(case))
        return (
            "Task: closed-ended medical VQA.\n"
            "Select exactly one official option label. Do not answer with option text.\n\n"
            f"{example_block}\n\n"
            f"Question:\n{str(case.get('question') or '').strip()}\n\n"
            f"Options:\n{options}\n\n"
            "Return only a JSON object. The values must be specific to this case. "
            "Use this schema: {\"reasoning_trace\": string, \"answer\": one_option_label}"
        )

    parts = [
        "Use the medical image and question to answer in strict JSON.",
        "Return keys `reasoning_trace` and `answer` only.",
    ]
    if task == "open" and prompt_style == "evidence_first":
        parts.extend(
            [
                "Use the image evidence first. Retrieved examples are analogies only.",
                "Do not copy a retrieved answer unless the same visual finding is present.",
                "Prefer concise findings tied to visible abnormality, location, morphology, or change.",
            ]
        )
    if task == "open" and prompt_style == "modality_guard":
        meta = case.get("metadata") or {}
        modality = str(case.get("modality") or meta.get("modality") or "").strip()
        organ = str(
            case.get("organ system") or case.get("organ_system") or meta.get("organ_system") or ""
        ).strip()
        anchor_parts = [p for p in [modality, organ] if p]
        if anchor_parts:
            parts.append("Case context: " + "; ".join(anchor_parts) + ".")
        parts.extend(
            [
                "Ground your answer in this case's modality and anatomical region only.",
                "Do not describe findings, structures, or pathologies outside this region.",
                "Retrieved examples are analogies only; reject any retrieved answer that describes a different modality or organ.",
            ]
        )
    if examples:
        blocks = []
        for idx, ex in enumerate(examples, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"Retrieved example {idx}:",
                        f"Question: {str(ex.get('question', '')).strip()}",
                        f"Known answer: {str(ex.get('answer', '')).strip()}",
                        f"Visual cue: {str(ex.get('visual_description', '')).strip()}",
                    ]
                )
            )
        parts.append("\n\n".join(blocks))
    parts.append(f"Question:\n{str(case.get('question') or '').strip()}")
    if task == "mcq":
        options = "\n".join(f"{opt['label']}. {opt['text']}" for opt in options_for_case(case))
        parts.extend(
            [
                "Select exactly one official option label.",
                f"Options:\n{options}",
            ]
        )
    else:
        parts.append("Provide a concise image-grounded answer.")
    return "\n\n".join(part for part in parts if part.strip())


def response_for_case(case: dict[str, Any], response_style: str = "visual_answer") -> str:
    answer = str(case.get("answer") or "").strip()
    visual = str(case.get("visual_description") or "").strip()
    if task_type(case) == "mcq" and response_style in {
        "mcq_answer_first",
        "mcq_option_answer_first",
    }:
        reasoning = "Selected the option best supported by the image."
        if response_style == "mcq_option_answer_first":
            reasoning = option_text(case, answer)
        return json.dumps(
            {
                "answer": answer,
                "reasoning_trace": reasoning,
            },
            ensure_ascii=False,
        )
    if not visual:
        visual = "The visual evidence supports the final answer."
    if task_type(case) == "open" and response_style == "concise_open":
        visual = "The image evidence supports the concise final answer."
    payload = {
        "reasoning_trace": visual,
        "answer": answer,
    }
    return json.dumps(payload, ensure_ascii=False)


def retrieval_examples(
    case: dict[str, Any],
    bank: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    if bank is None or top_k <= 0:
        return []
    query = " ".join(
        [
            str(case.get("question") or ""),
            " ".join(opt["text"] for opt in options_for_case(case)),
        ]
    )
    hits = bank.search(query, task_type=task_type(case), top_k=top_k + 3)
    examples: list[dict[str, Any]] = []
    case_id = str(case.get("case_id"))
    for hit in hits:
        ex = hit.example
        if str(ex.get("case_id")) == case_id:
            continue
        examples.append(ex)
        if len(examples) >= top_k:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL SFT data for MedReason Qwen2.5-VL LoRA.")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--train-img-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--exclude-case-files", type=Path, nargs="*", default=[])
    parser.add_argument("--retrieval-bank", type=Path, default=None)
    parser.add_argument("--top-k-examples", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-mcq", type=int, default=0)
    parser.add_argument("--max-open", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mcq-only", action="store_true")
    parser.add_argument("--open-only", action="store_true")
    parser.add_argument(
        "--mcq-option-permutations",
        type=int,
        default=0,
        help="Number of additional deterministic option permutations per MCQ case.",
    )
    parser.add_argument(
        "--response-style",
        choices=(
            "visual_answer",
            "concise_open",
            "mcq_answer_first",
            "mcq_option_answer_first",
        ),
        default="visual_answer",
        help="SFT response style. MCQ answer-first styles put the scored label first in the JSON target.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("baseline", "evidence_first", "modality_guard", "runtime_mcq"),
        default="baseline",
        help="SFT prompt style. runtime_mcq mirrors the strong-baseline MCQ inference prompt.",
    )
    args = parser.parse_args()

    bank = None
    if args.retrieval_bank:
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "docker" / "medreason"))
        from medreason_docker.retrieval import RetrievalBank

        bank = RetrievalBank.from_json(args.retrieval_bank)

    excluded = excluded_ids(args.exclude_case_files)
    cases = json.loads(args.train_json.read_text(encoding="utf-8"))["cases"]
    if args.shuffle:
        random.Random(args.seed).shuffle(cases)
    rows = []
    task_counts = {"mcq": 0, "open": 0}
    for case in cases:
        task = task_type(case)
        if str(case.get("case_id")) in excluded:
            continue
        if not str(case.get("answer") or "").strip():
            continue
        if args.mcq_only and task != "mcq":
            continue
        if args.open_only and task != "open":
            continue
        if args.max_mcq > 0 and task == "mcq" and task_counts["mcq"] >= args.max_mcq:
            continue
        if args.max_open > 0 and task == "open" and task_counts["open"] >= args.max_open:
            continue
        image_path = image_path_for_case(case, args.train_img_dir)
        if not Path(image_path).exists():
            continue
        variants = [case]
        if task == "mcq":
            variants.extend(
                permute_mcq_options(case, seed=args.seed, variant=variant)
                for variant in range(1, args.mcq_option_permutations + 1)
            )
        for variant_index, variant_case in enumerate(variants):
            examples = retrieval_examples(variant_case, bank=bank, top_k=args.top_k_examples)
            rows.append(
                {
                    "case_id": str(case["case_id"]),
                    "task_type": task,
                    "image_path": image_path,
                    "prompt": prompt_for_case(variant_case, examples, prompt_style=args.prompt_style),
                    "response": response_for_case(variant_case, response_style=args.response_style),
                    "answer": str(variant_case.get("answer") or "").strip(),
                    "option_permutation": variant_index,
                }
            )
            task_counts[task] += 1
            if args.max_examples > 0 and len(rows) >= args.max_examples:
                break
        if args.max_examples > 0 and len(rows) >= args.max_examples:
            break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"wrote {args.output_jsonl} examples={len(rows)} "
        f"counts={task_counts} excluded={len(excluded)}"
    )


if __name__ == "__main__":
    main()
