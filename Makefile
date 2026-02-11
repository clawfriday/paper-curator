SHELL := /bin/bash
PYTHON := python
VENV_DIR := .venv
ACTIVATE := source $(VENV_DIR)/bin/activate

.PHONY: install install-frontend test run clean docker-build docker-run \
        singularity-build singularity-run singularity-stop pull-slack \
        test-db-start test-db-stop test-db-init test-download-samples \
        validation functional integration

# Create virtual environment and install all dependencies
install: $(VENV_DIR)/bin/activate install-frontend

$(VENV_DIR)/bin/activate: pyproject.toml
	test -d $(VENV_DIR) || python -m venv $(VENV_DIR)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -e ".[dev]"
	touch $(VENV_DIR)/bin/activate

install-frontend:
	@echo "Installing frontend dependencies..."
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm is not installed"; exit 1; }
	cd src/frontend && npm install
	@echo "Frontend dependencies installed."

# =============================================================================
# Testing (assumes .venv is activated in tmux)
# =============================================================================
# Usage: make test [type]
#   make test                 - Run all tests (except integration)
#   make test validation      - Run validation tests only
#   make test functional      - Run functional tests only
#   make test integration     - Run integration tests (requires test DB)

TEST_DB_PORT := 5433
BACKEND_URL ?= http://localhost:3100
TEST_TYPE := $(word 2,$(MAKECMDGOALS))

test:
ifeq ($(TEST_TYPE),validation)
	@echo "=== Running validation tests ==="
	BACKEND_URL=$(BACKEND_URL) pytest tests/validation -v
else ifeq ($(TEST_TYPE),functional)
	@echo "=== Running functional tests ==="
	BACKEND_URL=$(BACKEND_URL) pytest tests/functional -v -s
else ifeq ($(TEST_TYPE),integration)
	@echo "=== Running integration tests ==="
	PGPORT=$(TEST_DB_PORT) PGDATABASE=paper_curator_test BACKEND_URL=$(BACKEND_URL) pytest tests/integration -v -s
else
	@echo "=== Running all tests (except integration and deprecated) ==="
	BACKEND_URL=$(BACKEND_URL) pytest tests -v --ignore=tests/integration --ignore=tests/deprecated
endif

validation functional integration:
	@:

test-db-start:
	@echo "Starting test database on port $(TEST_DB_PORT)..."
	@mkdir -p tests/storage/pgdata
	@if command -v singularity >/dev/null 2>&1; then \
		singularity instance start \
			--bind $$(pwd)/tests/storage/pgdata:/var/lib/postgresql/data \
			--env POSTGRES_USER=curator \
			--env POSTGRES_PASSWORD=curator123 \
			--env POSTGRES_DB=paper_curator_test \
			containers/pgvector.sif pgtest 2>/dev/null || echo "Test DB may already be running"; \
	elif command -v docker >/dev/null 2>&1; then \
		docker run -d --name paper-curator-test-db \
			-p $(TEST_DB_PORT):5432 \
			-v $$(pwd)/tests/storage/pgdata:/var/lib/postgresql/data \
			-e POSTGRES_USER=curator \
			-e POSTGRES_PASSWORD=curator123 \
			-e POSTGRES_DB=paper_curator_test \
			pgvector/pgvector:pg16 2>/dev/null || echo "Test DB may already be running"; \
	else \
		echo "Error: Neither singularity nor docker found"; exit 1; \
	fi
	@sleep 3
	@echo "Test database ready on port $(TEST_DB_PORT)"

test-db-stop:
	@singularity instance stop pgtest 2>/dev/null || true
	@docker stop paper-curator-test-db 2>/dev/null || true
	@docker rm paper-curator-test-db 2>/dev/null || true

test-db-init:
	PGPORT=$(TEST_DB_PORT) PGDATABASE=paper_curator_test python scripts/init_db.py

test-download-samples:
	@mkdir -p tests/storage/downloads
	@python -c 'import arxiv, os; \
		papers = ["1706.03762","1810.04805","2005.14165","1512.03385","1406.2661","1706.03741","2010.11929","2103.00020","2302.13971","2303.08774"]; \
		[print(f"Downloading {p}...") or next(arxiv.Search(id_list=[p]).results()).download_pdf(dirpath="tests/storage/downloads", filename=f"{p}.pdf") \
		 for p in papers if not os.path.exists(f"tests/storage/downloads/{p}.pdf")]'

# =============================================================================
# Development
# =============================================================================

connect:
	./scripts/connect_endpoint.sh

run:
	docker compose -f src/compose.yml up --build

run-local:
	cd src/backend && uvicorn app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker compose -f src/compose.yml build --no-cache

docker-stop:
	docker compose -f src/compose.yml down

# =============================================================================
# Singularity/Apptainer targets
# =============================================================================

CONTAINER_DIR := containers
SIF_BACKEND := $(CONTAINER_DIR)/backend.sif
SIF_FRONTEND := $(CONTAINER_DIR)/frontend.sif
SIF_DB := $(CONTAINER_DIR)/pgvector.sif
CONFIG_FILE := config/config.yaml

singularity-build:
	@mkdir -p $(CONTAINER_DIR)
	singularity pull --force $(SIF_DB) docker://pgvector/pgvector:pg16
	singularity build --force --fakeroot $(SIF_BACKEND) $(CONTAINER_DIR)/backend.def
	singularity build --force --fakeroot $(SIF_FRONTEND) $(CONTAINER_DIR)/frontend.def

singularity-build-backend:
	@mkdir -p $(CONTAINER_DIR)
	singularity build --force --fakeroot $(SIF_BACKEND) $(CONTAINER_DIR)/backend.def

singularity-run:
	./scripts/hpc-services.sh start

singularity-stop:
	./scripts/hpc-services.sh stop

# =============================================================================
# Slack Integration
# =============================================================================

SLACK_TOKEN_FILE := $(HOME)/.ssh/.slack

pull-slack:
	@if [ ! -f "$(SLACK_TOKEN_FILE)" ]; then \
		echo "Error: Slack token not found at $(SLACK_TOKEN_FILE)"; exit 1; \
	fi
	@SLACK_TOKEN=$$(cat "$(SLACK_TOKEN_FILE)") && \
	grep -A 100 "^slack:" "$(CONFIG_FILE)" | grep "^\s*-\s*http" | sed 's/^\s*-\s*//' | while read channel; do \
		echo "=== Processing: $$channel ==="; \
		curl -s -X POST http://localhost:3100/papers/batch-ingest \
			-H "Content-Type: application/json" \
			-d "{\"slack_channel\": \"$$channel\", \"slack_token\": \"$$SLACK_TOKEN\"}" | \
		python3 -c "import sys,json; d=json.load(sys.stdin); \
			print(f'Result: {d.get(\"success\",0)} success, {d.get(\"skipped\",0)} skipped, {d.get(\"errors\",0)} errors')"; \
	done;

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache
