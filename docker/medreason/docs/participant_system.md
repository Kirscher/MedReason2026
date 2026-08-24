# Implementing a MedReason System

The challenge asks participants to develop an MLLM system, not merely to upload a model checkpoint. The official Docker interface therefore exposes a generic `MedReasonSystem` abstraction.

A valid system may include:

- a single MLLM or VLM;
- a multi-model ensemble;
- a retrieval-augmented pipeline;
- an agentic workflow;
- a visual-evidence extraction stage;
- self-verification or answer revision;
- uncertainty-aware refusal or abstention;
- MCQ-specific and open-ended-specific modules.

All components must run inside the submitted Docker image during official evaluation.

## Required interface

```python
class MedReasonSystem:
    def setup(self) -> None:
        ...

    def predict_case(self, case: MedReasonCase) -> MedReasonPrediction:
        ...
```

`setup()` is called once before case inference. `predict_case()` is called once per case.

## Important constraints

- Do not require internet access at inference time.
- Do not call external APIs during official evaluation.
- Do not require manual interaction.
- Return exactly one prediction per input case.
- For MCQ, return an official option label.
- For open-ended reasoning, return both `reasoning_trace` and `answer`.
