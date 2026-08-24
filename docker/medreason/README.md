# MedReason inference container

This directory implements the offline container used in our MedReason 2026
submission. It routes MCQ and open-ended cases to separate Qwen2.5-VL-3B LoRA
adapters and provides option-aware TF-IDF retrieval.

## Container contract

The evaluator mounts `/input` and `/output`. The container must:

1. Read `/input/cases.json`.
2. Read all referenced image files under `/input`.
3. Write `/output/results.json`.
4. Return exactly one prediction for every input `case_id`.

No internet access or external API is required at inference time.

## Code layout

```text
process.py                                      # container entry point
medreason_docker/systems/strong_baseline_system.py # submitted pipeline
medreason_docker/retrieval.py                    # TF-IDF retrieval
medreason_docker/systems/smoke_system.py         # dependency-free smoke test
tools/validate_output.py                         # output validator
```

## Smoke test

```bash
MEDREASON_SYSTEM=smoke \
MEDREASON_INPUT_DIR=./test \
MEDREASON_OUTPUT_FILE=./output/results.json \
python3 process.py

python3 tools/validate_output.py output/results.json --input-json test/cases.json
```

```bash
./build.sh medreason-smoke
./test.sh medreason-smoke
```

## Full system

The repository does not distribute challenge data, Qwen weights, retrieval
banks, or LoRA weights. Stage authorized local copies before building:

```bash
models/Qwen2.5-VL/
retrieval/retrieval_bank.json
lora/final_improved_mcq/
lora/final_improved_open/
```

```bash
./build.sh medreason-camera-ready \
  --build-arg INSTALL_VLM_DEPS=1 \
  --build-arg DEFAULT_LORA_PATH=/opt/app/lora/final_improved_mcq \
  --build-arg DEFAULT_OPEN_LORA_PATH=/opt/app/lora/final_improved_open \
  --build-arg DEFAULT_VLM_OVERRIDE_CONFIDENCE_THRESHOLD=0.257 \
  --build-arg DEFAULT_MAX_NEW_TOKENS=384 \
  --build-arg DEFAULT_OPEN_PROMPT_STYLE=modality_guard \
  --build-arg DEFAULT_STRICT_RESOURCES=1
```

Run with the official mount contract:

```bash
docker run --rm --gpus all \
  -v <input-dir>:/input:ro \
  -v <output-dir>:/output \
  medreason-camera-ready
```

The camera-ready runtime uses three in-prompt retrieved examples, option-aware
`sum5` scoring, a retrieval-confidence threshold of `0.257`, greedy decoding,
and a 384-token budget. Environment variables can override these defaults; see
`medreason_docker/config.py`.

Validate the result:

```bash
python3 tools/validate_output.py \
  <output-dir>/results.json --input-json <input-dir>/cases.json
```
