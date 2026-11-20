# Dockerfile for Liga MX Pipeline
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy dependency files first (for cache)
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY src/ ./src/
COPY config.yaml ./
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Create directories for data, models, predictions, logs
RUN mkdir -p data/models predictions logs cache && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default command
CMD ["python", "-m", "liga_mx.cli.commands", "--help"]