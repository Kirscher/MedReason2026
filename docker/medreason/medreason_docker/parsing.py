from __future__ import annotations

import json
import re
from typing import Any

from .schema import MedReasonCase

_JSON_DECODER = json.JSONDecoder()


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    raw = text or ""
    for idx, char in enumerate(raw):
        if char != "{":
            continue
        try:
            obj, _end = _JSON_DECODER.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_reasoning_and_answer(text: str, case: MedReasonCase) -> tuple[str, str]:
    """Parse a model response into `(reasoning_trace, answer)`.

    The recommended model output is JSON, but this parser also accepts common
    free-text formats to make the template robust during development.
    """

    raw = (text or "").strip()
    obj = _try_parse_json_object(raw)
    if obj is not None:
        reasoning = str(obj.get("reasoning_trace", obj.get("rationale", ""))).strip()
        answer = str(obj.get("answer", obj.get("final_answer", ""))).strip()
        return reasoning, answer

    reasoning = raw
    answer = raw
    for marker in ["Final answer:", "Answer:", "ANSWER:", "Final:"]:
        if marker in raw:
            prefix, suffix = raw.rsplit(marker, 1)
            reasoning = prefix.strip()
            answer = suffix.strip()
            break
    return reasoning, answer


def normalize_mcq_answer(answer: str, case: MedReasonCase) -> str:
    labels = sorted(case.option_labels, key=lambda x: (len(x), x))
    clean = (answer or "").strip()
    if clean in case.option_labels:
        return clean

    labels_by_upper = {label.upper(): label for label in labels}

    # Match only explicit label forms. Avoid generic word-boundary matching
    # because the label "A" also matches the English article "a".
    for label in labels:
        escaped = re.escape(label)
        patterns = [
            rf"^\(?{escaped}\)?[\s\.:\-)]*$",
            rf"^\(?{escaped}\)?[\.:\-)]\s*.+$",
            rf"^(?:option|answer|final answer|final)\s*[:\-]?\s*\(?{escaped}\)?(?:\s|[\.\:\-\)]|$)",
        ]
        for pattern in patterns:
            if re.search(pattern, clean, flags=re.IGNORECASE):
                return label

    upper = clean.upper()
    if upper in labels_by_upper:
        return labels_by_upper[upper]

    # Fallback: if the model copied the option text exactly, map it back to the label.
    normalized_answer_text = re.sub(r"\s+", " ", clean.strip().strip(" .")).lower()
    for option in case.options:
        normalized_option_text = re.sub(r"\s+", " ", option.text.strip().strip(" .")).lower()
        if normalized_answer_text and normalized_answer_text == normalized_option_text:
            return option.label

    return clean
