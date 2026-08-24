from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

OPTION_LABELS = ("A", "B", "C", "D", "E")


def task_type(case: dict[str, Any]) -> str:
    raw = str(case.get("task_type") or case.get("question type") or "").strip().lower()
    return "open" if raw in {"open", "open-ended", "open_ended"} else "mcq"


def option_text(case: dict[str, Any], label: str) -> str:
    text = str(case.get(label) or "").strip()
    prefix = f"{label})"
    return text[len(prefix) :].strip() if text.startswith(prefix) else text


def case_image_paths(case: dict[str, Any], split_dir: Path, image_prefix: str = "imgs") -> list[str]:
    raw_values: list[str] = []
    if case.get("image_path"):
        raw_values.append(str(case["image_path"]))
    if isinstance(case.get("image_paths"), list):
        raw_values.extend(str(path) for path in case["image_paths"])

    output_paths: list[str] = []
    for raw in raw_values:
        path = Path(raw)
        if path.is_absolute():
            try:
                output_paths.append(path.relative_to(split_dir).as_posix())
                continue
            except ValueError:
                output_paths.append(f"{image_prefix}/{path.name}")
                continue
        if (split_dir / path).exists():
            output_paths.append(path.as_posix())
        elif path.parts and path.parts[0] == image_prefix:
            output_paths.append(path.as_posix())
        else:
            output_paths.append(f"{image_prefix}/{path.name}")
    return output_paths


def convert_case(case: dict[str, Any], split_dir: Path, image_prefix: str = "imgs") -> dict[str, Any]:
    image_paths = case_image_paths(case, split_dir=split_dir, image_prefix=image_prefix)
    item: dict[str, Any] = {
        "case_id": str(case["case_id"]),
        "task_type": task_type(case),
        "question": str(case.get("question") or ""),
        "metadata": {
            "original_image_path": case.get("image_path"),
            "original_image_paths": case.get("image_paths"),
            "question_type": case.get("question type"),
            "modality": case.get("modality") or case.get("primary_modality"),
            "organ_system": case.get("organ system") or case.get("organ_system"),
            "task": case.get("task"),
            "subtype": case.get("subtype"),
        },
    }
    if len(image_paths) == 1:
        item["image_path"] = image_paths[0]
    else:
        item["image_paths"] = image_paths
    options = [
        {"label": label, "text": option_text(case, label)}
        for label in OPTION_LABELS
        if case.get(label)
    ]
    if options:
        item["options"] = options
    return item


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
    parser = argparse.ArgumentParser(description="Prepare official MedReason cases.json for local Docker runs.")
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking imgs/.")
    parser.add_argument("--limit", type=int, default=0, help="Keep only the first N cases after reading the source JSON.")
    args = parser.parse_args()

    payload = json.loads(args.source_json.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.limit > 0:
        cases = cases[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy_images(args.split_dir, args.output_dir, copy=args.copy_images)
    out = {"cases": [convert_case(case, split_dir=args.split_dir) for case in cases]}
    out_path = args.output_dir / "cases.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} cases={len(out['cases'])}")


if __name__ == "__main__":
    main()
