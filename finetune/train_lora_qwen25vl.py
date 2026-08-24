from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, get_scheduler


SYSTEM_PROMPT = (
    "You are a medical visual reasoning assistant for the MedReason Challenge. "
    "Use the image evidence, question, and options when present. Return only valid JSON."
)


class JsonlDataset(Dataset):
    def __init__(self, path: Path, max_examples: int = 0, seed: int = 13) -> None:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        random.Random(seed).shuffle(rows)
        if max_examples > 0:
            rows = rows[:max_examples]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def load_model(model_path: Path, load_in_4bit: bool):
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    load_errors: list[str] = []
    for class_name in (
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        try:
            transformers_mod = __import__("transformers", fromlist=[class_name])
            model_cls = getattr(transformers_mod, class_name)
            return model_cls.from_pretrained(str(model_path), **kwargs)
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"{class_name}: {exc}")
    raise RuntimeError("Unable to load Qwen-compatible VLM:\n" + "\n".join(load_errors))


def build_inputs(processor, row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    user_content = [
        {"type": "image", "image": row["image_path"]},
        {"type": "text", "text": row["prompt"]},
    ]
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": row["response"]},
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(prompt_messages)
    full = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    labels = full["input_ids"].clone()
    prompt_len = min(prompt["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100
    if "attention_mask" in full:
        labels[full["attention_mask"] == 0] = -100
    full["labels"] = labels
    return {key: value.to(device) for key, value in full.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Qwen2.5-VL LoRA adapter for MedReason SFT.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("constant", "linear", "cosine"),
        default="constant",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this LoRA training script.")
    device = torch.device("cuda")

    dataset = JsonlDataset(args.train_jsonl, max_examples=args.max_examples, seed=args.seed)
    if not dataset:
        raise RuntimeError("No training examples found.")

    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = load_model(args.model_path, load_in_4bit=args.load_in_4bit)
    model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda rows: rows[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_updates = args.max_steps
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_updates,
    )
    epochs = max(1, math.ceil(total_updates * args.gradient_accumulation_steps / len(loader)))
    progress = tqdm(total=total_updates, desc="lora steps")
    optimizer.zero_grad(set_to_none=True)
    step = 0
    accum = 0
    running_loss = 0.0
    for _epoch in range(epochs):
        for row in loader:
            batch = build_inputs(processor, row, device=device)
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            accum += 1
            if accum % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.update(1)
                if step % 5 == 0:
                    progress.set_postfix(loss=running_loss / 5)
                    running_loss = 0.0
                if args.save_every > 0 and step % args.save_every == 0:
                    ckpt = args.output_dir / f"checkpoint-{step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)
                if step >= total_updates:
                    break
        if step >= total_updates:
            break
    progress.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    metadata = {
        "model_path": str(args.model_path),
        "train_jsonl": str(args.train_jsonl),
        "max_examples": args.max_examples,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "load_in_4bit": args.load_in_4bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
