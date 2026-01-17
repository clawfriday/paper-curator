#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
BASE_URL="${SERVICE_BASE_URL}"
ARXIV_ID="1706.03762"
OUTPUT_DIR="storage/downloads"
OUTPUT_JSON="storage/outputs/arxiv_download.json"

# Save output locally so downstream steps can reference the file paths.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --arxiv-id)
      ARXIV_ID="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
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

mkdir -p "$(dirname "${OUTPUT_JSON}")"
mkdir -p "${OUTPUT_DIR}"

# Download PDF/LaTeX for a single arXiv paper.
curl -f -sS -X POST "${BASE_URL}/arxiv/download" \
  -H "Content-Type: application/json" \
  -d "{\"arxiv_id\":\"${ARXIV_ID}\",\"output_dir\":\"${OUTPUT_DIR}\"}" \
  > "${OUTPUT_JSON}"

echo "Wrote response: ${OUTPUT_JSON}"
