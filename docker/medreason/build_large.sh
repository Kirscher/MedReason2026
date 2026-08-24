#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-medreason-local:vlm}"
if [ "$#" -gt 0 ]; then
  shift
fi

OUTPUT_TAR="${1:-../../artifacts/docker/${IMAGE_NAME//[:\/]/-}.tar}"
if [ "$#" -gt 0 ]; then
  shift
fi

BUILDER_NAME="${MEDREASON_BUILDX_BUILDER:-medreason-home}"

mkdir -p "$(dirname "${OUTPUT_TAR}")"

if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use --bootstrap
else
  docker buildx inspect "${BUILDER_NAME}" --bootstrap >/dev/null
fi

docker buildx build \
  --builder "${BUILDER_NAME}" \
  -t "${IMAGE_NAME}" \
  --output "type=docker,dest=${OUTPUT_TAR}" \
  "$@" \
  .

printf 'Wrote %s\n' "${OUTPUT_TAR}"
