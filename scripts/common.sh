#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-config/paperqa.yaml}"

get_config_value() {
  local key="$1"
  awk -F': ' -v k="$key" '$1==k {gsub(/"/, "", $2); print $2}' "${CONFIG_FILE}"
}

SERVICE_BASE_URL="$(get_config_value service_base_url)"
SERVICE_CONTAINER_NAME="$(get_config_value service_container_name)"
