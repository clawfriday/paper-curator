#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
BASE_URL="${SERVICE_BASE_URL}"

# Allow overriding the service URL without editing the script.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

# Probe service health to verify the server is reachable.
curl -f -sS "${BASE_URL}/health"
