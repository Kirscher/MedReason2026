from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

OPTION_LABELS = ("A", "B", "C", "D", "E")


def task_type(case: dict[str, Any]) -> str:
    raw = str(case.get("question type") or case.get("task_type") or "").strip().lower()
    return "open" if raw in {"open", "open-ended", "open_ended"} else "mcq"


def option_text(case: dict[str, Any], label: str) -> str:
    text = str(case.get(label) or "").strip()
    prefix = f"{label})"
    return text[len(prefix) :].strip() if text.startswith(prefix) else text


def to_case(case: dict[str, Any], image_prefix: str = "imgs") -> dict[str, Any]:
    image_name = Path(str(case.get("image_path") or "")).name
    item: dict[str, Any] = {
        "case_id": str(case["case_id"]),
        "task_type": task_type(case),
        "image_path": f"{image_prefix}/{image_name}",
        "question": str(case.get("question") or ""),
        "metadata": {
            "modality": case.get("modality") or case.get("primary_modality"),
            "organ_system": case.get("organ system"),
            "task": case.get("task"),
            "subtype": case.get("subtype"),
            "source_split_order": case.get("split_order"),
        },
    }
    options = [
        {"label": label, "text": option_text(case, label)}
        for label in OPTION_LABELS
        if case.get(label)
    ]
    if options:
        item["options"] = options
    return item


def to_ground_truth(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(case["case_id"]),
        "task_type": task_type(case),
        "answer": str(case.get("answer") or "").strip(),
    }


def link_or_copy_images(split_dir: Path, output_dir: Path, copy: bool) -> None:
    src = split_dir / "imgs"
    dst = output_dir / "imgs"
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copytree(src, dst)
    else:
        os.symlink(src, dst, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a labeled MedReason train holdout.")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-mcq", type=int, default=50)
    parser.add_argument("--num-open", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.train_json.read_text(encoding="utf-8"))
    cases = [case for case in payload["cases"] if str(case.get("answer") or "").strip()]
    mcq = [case for case in cases if task_type(case) == "mcq"]
    open_cases = [case for case in cases if task_type(case) == "open"]

    rng = random.Random(args.seed)
    selected = rng.sample(mcq, min(args.num_mcq, len(mcq))) + rng.sample(
        open_cases, min(args.num_open, len(open_cases))
    )
    rng.shuffle(selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy_images(args.split_dir, args.output_dir, copy=args.copy_images)
    (args.output_dir / "cases.json").write_text(
        json.dumps({"cases": [to_case(case) for case in selected]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "ground_truth.json").write_text(
        json.dumps({"answers": [to_ground_truth(case) for case in selected]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output_dir} cases={len(selected)} "
        f"mcq={sum(task_type(c) == 'mcq' for c in selected)} "
        f"open={sum(task_type(c) == 'open' for c in selected)}"
    )


if __name__ == "__main__":
    main()
