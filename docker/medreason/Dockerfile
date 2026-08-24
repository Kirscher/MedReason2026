ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

COPY requirements-vlm.txt /opt/app/requirements-vlm.txt
COPY requirements-vlm-runtime.txt /opt/app/requirements-vlm-runtime.txt
COPY requirements-awq.txt /opt/app/requirements-awq.txt
ARG INSTALL_VLM_DEPS=0
ARG INSTALL_TORCH_DEPS=1
ARG INSTALL_AWQ_DEPS=0
RUN set -eux; \
    if [ "$INSTALL_VLM_DEPS" = "1" ]; then \
      if [ "$INSTALL_TORCH_DEPS" = "1" ]; then \
        pip install --no-cache-dir -r /opt/app/requirements-vlm.txt; \
      else \
        pip install --no-cache-dir -r /opt/app/requirements-vlm-runtime.txt; \
      fi; \
    fi; \
    if [ "$INSTALL_AWQ_DEPS" = "1" ]; then \
      pip install --no-cache-dir -r /opt/app/requirements-awq.txt; \
    fi

ARG DEFAULT_LORA_PATH=
ARG DEFAULT_OPEN_LORA_PATH=
ARG DEFAULT_VLM_OVERRIDE_CONFIDENCE_THRESHOLD=0.257
ARG DEFAULT_MAX_NEW_TOKENS=384
ARG DEFAULT_OPEN_PROMPT_STYLE=modality_guard
ARG DEFAULT_STRICT_RESOURCES=0

ENV MEDREASON_INPUT_DIR=/input \
    MEDREASON_OUTPUT_DIR=/output \
    MEDREASON_OUTPUT_FILE=/output/results.json \
    MEDREASON_SYSTEM=strong_baseline \
    MEDREASON_MODEL_PATH=/opt/app/models/Qwen2.5-VL \
    MEDREASON_LORA_PATH=${DEFAULT_LORA_PATH} \
    MEDREASON_OPEN_LORA_PATH=${DEFAULT_OPEN_LORA_PATH} \
    MEDREASON_RETRIEVAL_BANK=/opt/app/retrieval/retrieval_bank.json \
    MEDREASON_MCQ_POLICY=hybrid_low_confidence \
    MEDREASON_OPTION_SCORE_MODE=sum5 \
    MEDREASON_VLM_OVERRIDE_CONFIDENCE_THRESHOLD=${DEFAULT_VLM_OVERRIDE_CONFIDENCE_THRESHOLD} \
    MEDREASON_MAX_NEW_TOKENS=${DEFAULT_MAX_NEW_TOKENS} \
    MEDREASON_OPEN_PROMPT_STYLE=${DEFAULT_OPEN_PROMPT_STYLE} \
    MEDREASON_STRICT_RESOURCES=${DEFAULT_STRICT_RESOURCES} \
    MEDREASON_LOG_EVERY=25

COPY . /opt/app
RUN chmod +x /opt/app/process.py

ENTRYPOINT ["python", "/opt/app/process.py"]
