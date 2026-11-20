# Makefile for Liga MX Pipeline

.PHONY: help install test lint format typecheck clean docker-build docker-run update predict train trade decay backtest fetch init

# Default target
help:
	@echo "Liga MX Pipeline - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install package in development mode"
	@echo "  make init          - Initialize project (dirs, download initial data)"
	@echo ""
	@echo "Development:"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run Ruff linter"
	@echo "  make format        - Format code with Ruff"
	@echo "  make typecheck     - Run MyPy type checker"
	@echo "  make clean         - Clean build artifacts"
	@echo ""
	@echo "Pipeline:"
	@echo "  make update        - Update master data, regenerate features"
	@echo "  make predict       - Generate predictions for upcoming fixtures"
	@echo "  make train         - Train all models"
	@echo "  make trade         - Run paper trading (place_all)"
	@echo "  make settle        - Settle paper trading bets"
	@echo "  make decay         - Run weekly decay detection"
	@echo "  make backtest      - Run monthly backtest"
	@echo "  make fetch         - Fetch data from all sources"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo ""

# Install package in development mode
install:
	pip install -e ".[dev]"

# Initialize project
init:
	python -m liga_mx.cli.commands init

# Run tests
test:
	pytest --cov=liga_mx --cov-report=term-missing --cov-fail-under=80

# Lint
lint:
	ruff check src/ tests/ scripts/

# Format
format:
	ruff format src/ tests/ scripts/

# Type check
typecheck:
	mypy src/liga_mx

# Clean
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Update pipeline (download data, rebuild master, regenerate features)
update:
	python -m liga_mx.cli.commands update

# Predict
predict:
	python -m liga_mx.cli.commands predict --market btts

# Train models
train:
	python -m liga_mx.cli.commands train --model all

# Paper trading
trade:
	python -m liga_mx.cli.commands trade place_all

settle:
	python -m liga_mx.cli.commands trade settle_all

# Monitoring
decay:
	python scripts/cron_decay.py

backtest:
	python scripts/cron_monthly.py

# Fetch data
fetch:
	python -m liga_mx.cli.commands fetch all

# Docker
docker-build:
	docker build -t liga-mx-pipeline:latest .

docker-run:
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/predictions:/app/predictions \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/.env:/app/.env:ro \
		liga-mx-pipeline:latest

# Full CI pipeline locally
ci: lint typecheck test

# Quick checks
check: lint typecheck