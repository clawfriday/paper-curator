#!/usr/bin/env bash
set -euo pipefail

# Start GROBID service in the background.
export GROBID_HOME="/opt/grobid/grobid-home"
export GROBID_CONFIG="/opt/grobid/grobid-home/config/grobid.yaml"
(
  cd /opt/grobid
  /opt/grobid/grobid-service/bin/grobid-service &
)

# Start FastAPI app in the background.
cd /app
python3 -m uvicorn services.app:app --host 0.0.0.0 --port 9000 &

# Run nginx in the foreground to keep the container alive.
nginx -g "daemon off;"
