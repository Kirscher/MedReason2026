# MedReason Docker I/O Schema

## Input

The container reads:

```text
/input/cases.json
```

`cases.json` may contain either a raw list of cases or an object with a `cases` list.

Example:

```json
{
  "cases": [
    {
      "case_id": "case_mcq_001",
      "task_type": "mcq",
      "image_path": "images/case_mcq_001.png",
      "question": "Which finding is most consistent with the image?",
      "options": [
        {"label": "A", "text": "Normal study"},
        {"label": "B", "text": "Cardiomegaly"},
        {"label": "C", "text": "Pneumothorax"}
      ]
    },
    {
      "case_id": "case_open_001",
      "task_type": "open",
      "image_paths": ["images/case_open_001.png"],
      "question": "Describe the main abnormality."
    }
  ]
}
```

`task_type` aliases such as `closed_ended` and `open_ended` are accepted internally, but the normalized output uses `mcq` and `open`.

## Output

The container writes:

```text
/output/results.json
```

Example:

```json
{
  "name": "MedReason system submission",
  "type": "Medical visual reasoning",
  "answers": [
    {
      "case_id": "case_mcq_001",
      "task_type": "mcq",
      "answer": "B",
      "reasoning_trace": "The image evidence supports option B."
    },
    {
      "case_id": "case_open_001",
      "task_type": "open",
      "reasoning_trace": "The image shows ...",
      "answer": "There is ..."
    }
  ],
  "version": {"major": 1, "minor": 0}
}
```

For open-ended cases, `reasoning_trace` and `answer` are both required.
