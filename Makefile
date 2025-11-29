# --- Docker Registry Configuration ---
DOCKER_REGISTRY := ghcr.io
# Force username to lowercase to satisfy Docker/GHCR requirements
DOCKER_USERNAME := $(shell echo "NoeFontana" | tr '[:upper:]' '[:lower:]')
DOCKER_IMAGE    := cs336-dev
DOCKER_TAG      ?= latest

# Helper variable for the full image reference
DOCKER_REF      := $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE)

.PHONY: help install build-native test lint check lint-check type-check clean docs docs-serve docs-build pre-commit

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and all dependencies
	uv sync --all-groups

build-native:
	CARGO_BUILD_FLAGS="--release" uv pip install -e .

test: ## Run tests
	USE_NATIVE_MERGE=1 uv run pytest

test-cov: ## Run tests with coverage report
	USE_NATIVE_MERGE=1 uv run pytest --cov --cov-report=html --cov-report=term

lint: ## Run linting checks
	uv run ruff check  --fix

format: ## Format code
	uv run ruff format

lint-check: ## Run linting checks and fix issues
	uv run ruff check

format-check: ## Check code formatting
	uv run ruff format --check

type-check: ## Run type checking
	uv run ty check

check: lint format-check type-check test ## Run all checks (lint, format, type-check, test)

clean: ## Clean up build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs: docs-serve ## Serve documentation locally (alias for docs-serve)

docs-serve: ## Serve documentation locally with auto-reload
	uv run mkdocs serve

docs-build: ## Build documentation
	uv run mkdocs build

pre-commit: ## Set up pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

update: ## Update all dependencies
	uv sync --upgrade

profile-tokenization: ## Profile the BPE tokenizer training script
	USE_NATIVE_MERGE=1 PYTHONPATH=src uv run scalene \
		--cpu --web --profile-exclude threading.py \
		--- \
		scripts/a1/train_bpe.py \
		--vocab-size 32000 \
		--input-path ~/datasets/cs336/owt_train.txt \
		--output-prefix ./owt

train-owt-tokenizer:
	USE_NATIVE_MERGE=1 PYTHONPATH=src uv run scripts/a1/train_bpe.py \
		--vocab-size 32000 \
		--input-path ~/datasets/cs336/owt_train.txt \
		--output-prefix ./results/owt

tokenize-dataset: ## Tokenize the default dataset using the default tokenizer
	PYTHONPATH=src uv run scripts/a1/tokenize_dataset.py

docker-build: ## Build the Docker image locally
	@echo "🛠️  Building Docker image: $(DOCKER_IMAGE):latest..."
	docker build -t $(DOCKER_IMAGE):latest .

docker-login-check: # Internal check (hidden from help)
	@echo "🔍 Verifying login to $(DOCKER_REGISTRY)..."
	@grep -q "$(DOCKER_REGISTRY)" ~/.docker/config.json || (echo "❌ Error: You are not logged into $(DOCKER_REGISTRY). Run 'docker login $(DOCKER_REGISTRY)' first." && exit 1)

docker-push: docker-login-check ## Tag and push the image to GHCR
	@echo "🏷️  Tagging image as: $(DOCKER_REF):$(DOCKER_TAG)"
	docker tag $(DOCKER_IMAGE):latest $(DOCKER_REF):$(DOCKER_TAG)

	@echo "🚀 Pushing to $(DOCKER_REGISTRY)..."
	docker push $(DOCKER_REF):$(DOCKER_TAG)

	@if [ "$(DOCKER_TAG)" != "latest" ]; then \
		echo "🏷️  Also pushing 'latest' tag for convenience..."; \
		docker tag $(DOCKER_IMAGE):latest $(DOCKER_REF):latest; \
		docker push $(DOCKER_REF):latest; \
	fi
	@echo "✅ Pushed successfully: https://github.com/NoeFontana/$(DOCKER_IMAGE)/pkgs/container/$(DOCKER_IMAGE)"

docker-release: docker-build docker-push ## Build, tag, and push in one step
