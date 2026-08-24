from __future__ import annotations

from .schema import MedReasonCase


def build_prompt(case: MedReasonCase) -> str:
    if case.task_type == "mcq":
        return build_mcq_prompt(case)
    return build_open_prompt(case)


def build_mcq_prompt(case: MedReasonCase) -> str:
    option_lines = "\n".join(f"{opt.label}. {opt.text}" for opt in case.options)
    return (
        "You are solving a clinically grounded medical visual reasoning question.\n"
        "Use the provided medical image evidence and the question.\n"
        "Select exactly one option label.\n\n"
        f"Question: {case.question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Return your response in JSON with keys `reasoning_trace` and `answer`. "
        "The `answer` value must be only the option label."
    )


def build_open_prompt(case: MedReasonCase) -> str:
    return (
        "You are solving a clinically grounded medical visual reasoning question.\n"
        "Use only the visual evidence that is available in the medical image and avoid unsupported claims.\n"
        "If the image evidence is insufficient, state the limitation explicitly.\n\n"
        f"Question: {case.question}\n\n"
        "Return your response in JSON with keys `reasoning_trace` and `answer`.\n"
        "The `reasoning_trace` should describe the image-grounded evidence used to answer.\n"
        "The `answer` should be a concise final answer."
    )
