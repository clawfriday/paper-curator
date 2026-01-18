#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
BASE_URL="${SERVICE_BASE_URL}"
EXTRACT_JSON="storage/outputs/extract.json"
OUTPUT_JSON="storage/outputs/paperqa_summary.json"
PDF_PATH=""

# Call summarize using a PDF path or extracted text.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --extract-json)
      EXTRACT_JSON="$2"
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

mkdir -p "$(dirname "${OUTPUT_JSON}")"

if [[ -n "${PDF_PATH}" ]]; then
  TEXT_PAYLOAD="{\"pdf_path\":\"${PDF_PATH}\"}"
else
  if [[ ! -f "${EXTRACT_JSON}" ]]; then
    echo "Missing extract JSON: ${EXTRACT_JSON}"
    exit 1
  fi
  # Build the request body using a standalone Python helper.
  TEXT_PAYLOAD="$(
    docker exec "${SERVICE_CONTAINER_NAME}" \
      python /app/services/scripts/build_text_payload.py \
      --extract-json "/app/${EXTRACT_JSON#./}"
  )"
fi

curl -f -sS -X POST "${BASE_URL}/summarize" \
  -H "Content-Type: application/json" \
  -d "${TEXT_PAYLOAD}" \
  > "${OUTPUT_JSON}"

echo "Wrote response: ${OUTPUT_JSON}"
