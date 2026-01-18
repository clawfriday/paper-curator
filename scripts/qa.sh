#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
BASE_URL="${SERVICE_BASE_URL}"
CONTEXT_FILE=""
QUESTION=""
OUTPUT_JSON="storage/outputs/paperqa_qa.json"
PDF_PATH=""

# Provide either a context file or a PDF path, plus a question.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --context-file)
      CONTEXT_FILE="$2"
      shift 2
      ;;
    --question)
      QUESTION="$2"
      shift 2
      ;;
    --output-json)
      OUTPUT_JSON="$2"
      shift 2
      ;;
    --pdf-path)
      PDF_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${QUESTION}" ]]; then
  echo "--question is required"
  exit 1
fi

if [[ -z "${PDF_PATH}" && -z "${CONTEXT_FILE}" ]]; then
  echo "--context-file or --pdf-path is required"
  exit 1
fi

if [[ -n "${CONTEXT_FILE}" && ! -f "${CONTEXT_FILE}" ]]; then
  echo "Missing context file: ${CONTEXT_FILE}"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSON}")"

if [[ -n "${PDF_PATH}" ]]; then
  PAYLOAD="{\"pdf_path\":\"${PDF_PATH}\",\"question\":\"${QUESTION}\"}"
else
  # Build the request body using a standalone Python helper.
  PAYLOAD="$(
    docker exec "${SERVICE_CONTAINER_NAME}" \
      python /app/services/scripts/build_qa_payload.py \
      --context-file "/app/${CONTEXT_FILE#./}" \
      --question "${QUESTION}"
  )"
fi

curl -f -sS -X POST "${BASE_URL}/qa" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" \
  > "${OUTPUT_JSON}"

echo "Wrote response: ${OUTPUT_JSON}"
