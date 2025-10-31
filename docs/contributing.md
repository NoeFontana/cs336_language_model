# Contributing

Thank you for your interest in contributing to this project!

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/NoeFontana/python-template.git
   cd python-template
   ```

3. Install dependencies:

   ```bash
   uv sync --all-groups
   ```

4. Set up pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Making Changes

1. Create a new branch for your feature:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and write tests

3. Run the test suite:

   ```bash
   make check
   ```

4. Commit your changes:

   ```bash
   git commit -m "Add your feature"
   ```

5. Push to your fork and create a pull request

## Code Style

This project uses:

- **ruff** for linting and formatting
- **pyright** for type checking
- **pytest** for testing

All checks must pass before merging.
