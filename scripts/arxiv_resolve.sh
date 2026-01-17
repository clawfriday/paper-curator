#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
BASE_URL="${SERVICE_BASE_URL}"
ARXIV_ID="1706.03762"
OUTPUT_JSON="storage/outputs/arxiv_resolve.json"

# Keep inputs overridable so the script is reusable for other papers.
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

# Resolve metadata for a single arXiv ID.
curl -f -sS -X POST "${BASE_URL}/arxiv/resolve" \
  -H "Content-Type: application/json" \
  -d "{\"arxiv_id\":\"${ARXIV_ID}\"}" \
  > "${OUTPUT_JSON}"

echo "Wrote response: ${OUTPUT_JSON}"
