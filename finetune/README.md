# MedReason LoRA fine-tuning

These scripts build task-specific SFT datasets and train QLoRA adapters for
Qwen2.5-VL-3B-Instruct. Only authorized local copies of the challenge data are
used; data and trained weights are not included in this repository.

Set local paths once:

```bash
export MODEL_PATH=<qwen-model-dir>
export TRAIN_JSON=<train-json>
export TRAIN_IMG_DIR=<train-image-dir>
export EXCLUDE_CASES=<development-cases-json>
```

The camera-ready MCQ adapter uses 4,096 option-permuted examples, one retrieved
example per training prompt, answer-first targets, seed 81, and a cosine
schedule. Build its dataset and train it with:

```bash
python3 finetune/build_sft_dataset.py \
  --train-json "$TRAIN_JSON" \
  --train-img-dir "$TRAIN_IMG_DIR" \
  --exclude-case-files "$EXCLUDE_CASES" \
  --retrieval-bank artifacts/retrieval_bank.json \
  --top-k-examples 1 \
  --shuffle --seed 81 --mcq-only \
  --mcq-option-permutations 1 \
  --max-examples 4096 \
  --prompt-style runtime_mcq \
  --response-style mcq_answer_first \
  --output-jsonl artifacts/finetune/mcq-camera-ready.jsonl
```

```bash
python3 finetune/train_lora_qwen25vl.py \
  --model-path "$MODEL_PATH" \
  --train-jsonl artifacts/finetune/mcq-camera-ready.jsonl \
  --output-dir artifacts/finetune/mcq-camera-ready \
  --max-examples 4096 \
  --max-steps 250 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --lr-scheduler-type cosine \
  --warmup-steps 20 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --seed 81 \
  --load-in-4bit
```

The open-ended adapter uses 1,024 examples with modality-guard prompting, one
retrieved example, seed 72, and 200 steps:

```bash
python3 finetune/build_sft_dataset.py \
  --train-json "$TRAIN_JSON" \
  --train-img-dir "$TRAIN_IMG_DIR" \
  --exclude-case-files "$EXCLUDE_CASES" \
  --retrieval-bank artifacts/retrieval_bank.json \
  --top-k-examples 1 \
  --shuffle --seed 72 --open-only \
  --max-examples 1024 \
  --prompt-style modality_guard \
  --response-style visual_answer \
  --output-jsonl artifacts/finetune/open-camera-ready.jsonl
```

```bash
python3 finetune/train_lora_qwen25vl.py \
  --model-path "$MODEL_PATH" \
  --train-jsonl artifacts/finetune/open-camera-ready.jsonl \
  --output-dir artifacts/finetune/open-camera-ready \
  --max-examples 1024 \
  --max-steps 200 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --seed 72 \
  --load-in-4bit
```

All development holdout identifiers must be passed to dataset construction and
excluded from the retrieval bank. This prevents exact case-ID retrieval, but it
does not by itself prevent the same source image from appearing under another
identifier; see the limitations in the paper.
