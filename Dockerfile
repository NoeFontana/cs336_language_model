# syntax=docker/dockerfile:1
# The line above enables specific BuildKit features like cache mounts

# 1. Base Stage: System Dependencies
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    pkg-config \
    clang \
    lld \
    ca-certificates

# 2. Tools Stage: Fetch uv and Rust
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# 3. Development Stage
FROM base AS dev

# Create user first so they own the environment
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o user && useradd -m -u $UID -g $GID -o -s /bin/bash user

# Set environment variables for uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=true \
    UV_PYTHON=3.11 \
    CARGO_BUILD_FLAGS="--release"

# Switch to user to ensure venv is owned by them
USER user
WORKDIR /app

# Copy only dependency definitions first
COPY --chown=user:user pyproject.toml uv.lock ./

# Install dependencies (and Python 3.11 automatically via uv)
# --frozen: strictly respects lockfile
# --no-install-project: installs deps but not the app itself (better caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy the rest of the source code
COPY --chown=user:user . .

# Install the project itself (editable mode is usually default for dev, but good to be explicit if not)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Set runtime environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    USE_NATIVE_MERGE=1 \
    CARGO_BUILD_FLAGS="--release"

# Keep container alive
CMD ["tail", "-f", "/dev/null"]
