SHELL := /bin/bash
PYTHON := /scratch/Projects/SPEC-SF-AISG/envs/infer/bin/python
VENV_DIR := .venv
ACTIVATE := source $(VENV_DIR)/bin/activate

.PHONY: install install-frontend test run clean docker-build docker-run \
       singularity-build singularity-run singularity-stop pull-slack

# Create virtual environment and install all dependencies
install: $(VENV_DIR)/bin/activate install-frontend

$(VENV_DIR)/bin/activate: pyproject.toml
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -e ".[dev]"
	touch $(VENV_DIR)/bin/activate

# Install frontend npm dependencies
install-frontend:
	@echo "Installing frontend dependencies..."
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm is not installed. Please install Node.js (https://nodejs.org/)"; exit 1; }
	cd src/frontend && npm install
	@echo "Frontend dependencies installed."

# Run pytest tests
test: install
	pytest tests -v

# Connect to the endpoints
connect:
	./scripts/connect_endpoint.sh

# Start docker-compose stack (frontend + backend)
run:
	docker compose -f src/compose.yml up --build

# Run backend locally with uvicorn (for development)
# Assumes .venv is already activated
run-local:
	cd src/backend && uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Build docker images without starting
docker-build:
	docker compose -f src/compose.yml build --no-cache

# Stop and remove containers
docker-stop:
	docker compose -f src/compose.yml down

# =============================================================================
# Singularity/Apptainer targets for HPC environments (Docker alternative)
# Ports are read from config/config.yaml (server.frontend_port, server.backend_port)
# =============================================================================

CONTAINER_DIR := containers
SIF_BACKEND := $(CONTAINER_DIR)/backend.sif
SIF_FRONTEND := $(CONTAINER_DIR)/frontend.sif
SIF_DB := $(CONTAINER_DIR)/pgvector.sif
CONFIG_FILE := config/config.yaml

# Build all Singularity containers
singularity-build:
	@echo "Building Singularity containers..."
	@mkdir -p $(CONTAINER_DIR)
	@echo "Pulling pgvector database container..."
	singularity pull --force $(SIF_DB) docker://pgvector/pgvector:pg16
	@echo "Building backend container..."
	singularity build --force --fakeroot $(SIF_BACKEND) $(CONTAINER_DIR)/backend.def
	@echo "Building frontend container..."
	singularity build --force --fakeroot $(SIF_FRONTEND) $(CONTAINER_DIR)/frontend.def
	@echo "All Singularity containers built successfully"

# Start all services using Singularity (reads ports from config/config.yaml)
singularity-run:
	@echo "Starting Paper Curator services with Singularity..."
	./scripts/hpc-services.sh start

# Stop all Singularity instances
singularity-stop:
	./scripts/hpc-services.sh stop

# Pull papers from Slack channels configured in config/config.yaml
# Token is read from ~/.ssh/.slack
SLACK_TOKEN_FILE := $(HOME)/.ssh/.slack
CONFIG_FILE := config/config.yaml

pull-slack:
	@if [ ! -f "$(SLACK_TOKEN_FILE)" ]; then \
		echo "Error: Slack token not found at $(SLACK_TOKEN_FILE)"; \
		echo "Create the file with your Slack User OAuth Token (xoxp-...)"; \
		exit 1; \
	fi
	@SLACK_TOKEN=$$(cat "$(SLACK_TOKEN_FILE)") && \
	echo "=== Pulling from all Slack channels in config.yaml ==="; \
	grep -A 100 "^slack:" "$(CONFIG_FILE)" | grep "^\s*-\s*http" | sed 's/^\s*-\s*//' | while read channel; do \
		echo ""; \
		echo "=== Processing channel: $$channel ==="; \
		curl -s -X POST http://localhost:3100/papers/batch-ingest \
			-H "Content-Type: application/json" \
			-d "{\"slack_channel\": \"$$channel\", \"slack_token\": \"$$SLACK_TOKEN\"}" | \
		python3 -c "import sys,json; d=json.load(sys.stdin); \
			print('\\n'.join(d.get('progress_log',['No progress log'])[-20:])); \
			print(f'\\n=== Result: {d.get(\"success\",0)} success, {d.get(\"skipped\",0)} skipped, {d.get(\"errors\",0)} errors ===')"; \
	done;

# =============================================================================
# End Singularity targets
# =============================================================================

# Clean up virtual environment and cached files
clean:
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache
