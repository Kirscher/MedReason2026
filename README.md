# Option-Aware Retrieval and Task-Specific VLM Adaptation for Medical VQA

[![Tests](https://github.com/Kirscher/MedReason2026/actions/workflows/tests.yml/badge.svg)](https://github.com/Kirscher/MedReason2026/actions/workflows/tests.yml)

Official implementation of our [MedReason 2026 paper](https://openreview.net/forum?id=0vvODUp46Q)
([arXiv preprint](https://arxiv.org/abs/2609.15530)).
The offline pipeline uses Qwen2.5-VL-3B-Instruct, task-specific LoRA adapters,
and an option-aware TF-IDF retrieval prior for medical visual question
answering.

## Results

| MCQ setting | Accuracy on H220 |
| --- | ---: |
| Nearest-label retrieval | 20.0% |
| Option-aware retrieval | 57.5% |
| Final MCQ adapter, `k=0` / `k=1` retrieved examples | 93.5% |
| Final MCQ adapter, `k=3` | 94.0% |
| Submitted confidence gate, `k=3` | 94.0% |

The submitted system obtained **93.20% MCQ accuracy** in the organizer's
official pre-evaluation. H220 is a repeatedly used development holdout, not an
independent test set; case-ID exclusion also does not preclude image overlap.
The paper reports the full ablations and the preliminary 20-case open-ended
analysis.

## Repository

- `docker/medreason/`: offline inference container and submitted pipeline.
- `finetune/`: QLoRA dataset construction and training.
- `scripts/`: retrieval, calibration, evaluation, and ablation utilities.
- `medreason_baseline/`: lightweight retrieval-only baseline.

Challenge data, model weights, retrieval banks, LoRA weights, and evaluation
exports are not redistributed in this repository.

## Quick start

```bash
cd docker/medreason
./build.sh medreason-smoke
./test.sh medreason-smoke
```

Build a retrieval bank from an authorized local copy of the training split:

```bash
python3 scripts/build_retrieval_bank.py \
  --train-json <train.json> --output artifacts/retrieval_bank.json
```

Run the full pipeline after providing the backbone and adapters locally:

```bash
env PYTHONPATH=docker/medreason \
  MEDREASON_SYSTEM=strong_baseline \
  MEDREASON_INPUT_DIR=<input-dir> \
  MEDREASON_OUTPUT_DIR=<output-dir> \
  MEDREASON_MODEL_PATH=<qwen-model-dir> \
  MEDREASON_LORA_PATH=<mcq-adapter-dir> \
  MEDREASON_OPEN_LORA_PATH=<oe-adapter-dir> \
  MEDREASON_RETRIEVAL_BANK=artifacts/retrieval_bank.json \
  MEDREASON_MCQ_POLICY=hybrid_low_confidence \
  MEDREASON_TOP_K_EXAMPLES=3 \
  MEDREASON_VLM_OVERRIDE_CONFIDENCE_THRESHOLD=0.257 \
  MEDREASON_OPEN_PROMPT_STYLE=modality_guard \
  MEDREASON_MAX_NEW_TOKENS=384 \
  python3 docker/medreason/process.py
```

See [`docker/medreason/README.md`](docker/medreason/README.md) for the container
contract and [`finetune/README.md`](finetune/README.md) for the camera-ready
training configuration.

## Citation

If you use this work, please cite the preprint:

```bibtex
@misc{kirscher2026optionawareretrievaltaskspecificvlm,
      title={Option-Aware Retrieval and Task-Specific VLM Adaptation for Medical VQA},
      author={Tristan Kirscher and Niklas C. Koser and Soren Pirk},
      year={2026},
      eprint={2609.15530},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.15530},
}
```

## License

The repository code is released under the [Apache License 2.0](LICENSE).
This license does not apply to MedReason challenge data, model weights, or
third-party resources, which are not distributed here.
