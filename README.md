# Python Template

[![CI](https://github.com/NoeFontana/python-template/workflows/CI/badge.svg)](https://github.com/NoeFontana/python-template/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

A modern Python project template with best practices and tooling.

## Features

- 🚀 **Modern tooling**: Uses [uv](https://docs.astral.sh/uv/) for dependency management, [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [pyright](https://microsoft.github.io/pyright/) for type checking
- 📦 **Src layout**: Follows the recommended [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) for better packaging
- 🧪 **Testing**: Comprehensive testing setup with [pytest](https://docs.pytest.org/), coverage reporting, and parallel test execution
- 🔄 **CI/CD**: GitHub Actions workflows for testing, linting, type checking, and security scanning
- 📚 **Documentation**: Ready-to-use [MkDocs](https://www.mkdocs.org/) configuration with Material theme
- 🔧 **Pre-commit hooks**: Automated code quality checks with [pre-commit](https://pre-commit.com/)
- 🏷️ **Type hints**: Full type annotation support with static type checking
- 📋 **Code quality**: Enforced code standards with comprehensive linting rules
- 🔐 **Security**: Automated vulnerability scanning with safety checks

## Requirements

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/NoeFontana/python-template.git
cd python-template

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv sync

# Install development dependencies
uv sync --all-groups
```

## Usage

### Development

```bash
# Activate the virtual environment (if using pip)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run tests
uv run pytest
# or with pip: pytest

# Run tests with coverage
uv run pytest --cov
# or with pip: pytest --cov

# Lint and format code
uv run ruff check .          # Check for issues
uv run ruff check . --fix    # Fix auto-fixable issues
uv run ruff format .         # Format code

# Type checking
uv run pyright

# Run all checks (lint, format, type check, test)
make check  # If using the provided Makefile
```

### Pre-commit hooks

Set up pre-commit hooks to automatically run code quality checks:

```bash
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

### Documentation

Build and serve documentation locally:

```bash
# Install docs dependencies
uv sync --group docs

# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```

## Project Structure

```
python-template/
├── .github/
│   ├── workflows/          # GitHub Actions CI/CD
│   └── dependabot.yml      # Dependabot configuration
├── docs/                   # Documentation source
├── src/
│   └── python_template/    # Main package source code
│       ├── __init__.py
│       └── ...
├── tests/                  # Test suite
│   ├── __init__.py
│   └── ...
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile               # Development shortcuts
├── pyproject.toml         # Project configuration
└── README.md
```

## Configuration

The project uses `pyproject.toml` for all tool configuration:

- **Build system**: Uses [hatchling](https://hatch.pypa.io/) as the build backend
- **Dependencies**: Managed by uv, defined in `pyproject.toml`
- **Linting & Formatting**: Configured for ruff with sensible defaults
- **Type Checking**: Pyright configuration with strict settings
- **Testing**: Pytest with coverage reporting and useful plugins
- **Documentation**: MkDocs with Material theme and Python docstring support

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite (`uv run pytest`)
5. Run code quality checks (`uv run ruff check . && uv run ruff format . && uv run pyright`)
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow [PEP 8](https://pep8.org/) style guidelines (enforced by ruff)
- Write comprehensive tests for new functionality
- Add type hints to all public APIs
- Update documentation for user-facing changes
- Keep the changelog updated

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [uv](https://docs.astral.sh/uv/) for fast Python package management
- [ruff](https://docs.astral.sh/ruff/) for lightning-fast Python linting and formatting
- [pyright](https://microsoft.github.io/pyright/) for static type checking
- [pytest](https://docs.pytest.org/) for the testing framework
- [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for documentation
