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

build: ## Build the package
	uv build

publish: ## Publish the package to PyPI (requires PYPI_TOKEN)
	uv publish

update: ## Update all dependencies
	uv sync --upgrade

profile-tokenization: ## Profile the BPE tokenizer training script
	USE_NATIVE_MERGE=1 PYTHONPATH=src uv run scalene \
		--cpu --web --profile-exclude threading.py \
		--- \
		src/cs336/train_bpe.py \
		--vocab-size 32000 \
		--input-path ~/datasets/cs336/owt_train.txt \
		--output-prefix ./owt
