#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
BASE_URL="${SERVICE_BASE_URL}"
PDF_PATH=""
OUTPUT_JSON="storage/outputs/extract.json"

# Require an explicit PDF path to keep this script single-endpoint focused.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --pdf-path)
      PDF_PATH="$2"
      shift 2
      ;;
    --output-json)
      OUTPUT_JSON="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${PDF_PATH}" ]]; then
  echo "--pdf-path is required"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSON}")"

# Extract TEI + sections + references from the PDF via GROBID.
curl -f -sS -X POST "${BASE_URL}/pdf/extract" \
  -H "Content-Type: application/json" \
  -d "{\"pdf_path\":\"${PDF_PATH}\"}" \
  > "${OUTPUT_JSON}"

echo "Wrote response: ${OUTPUT_JSON}"
