#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-medreason-docker:latest}"
HOST_OUTPUT_DIR="${2:-$(pwd)/output}"
PYTHON_BIN="${PYTHON:-python3}"
mkdir -p "${HOST_OUTPUT_DIR}"

TEST_SYSTEM="${MEDREASON_TEST_SYSTEM:-smoke}"

docker run --rm \
  -e MEDREASON_SYSTEM="${TEST_SYSTEM}" \
  -e MEDREASON_DISABLE_VLM="${MEDREASON_DISABLE_VLM:-0}" \
  -e MEDREASON_STRICT_RESOURCES="${MEDREASON_STRICT_RESOURCES:-0}" \
  -v "$(pwd)/test:/input:ro" \
  -v "${HOST_OUTPUT_DIR}:/output" \
  "${IMAGE_NAME}"

"${PYTHON_BIN}" tools/validate_output.py "${HOST_OUTPUT_DIR}/results.json" --input-json test/cases.json

echo "Smoke test completed successfully. Output: ${HOST_OUTPUT_DIR}/results.json"
