.PHONY: help install install-dev lint format type-check test test-unit test-integration \
        test-regression reproduce generate-data run-simulation run-evaluation \
        generate-figures generate-tables docs docker-build docker-up docker-down clean

# Default target
help:
	@echo "FedAgent-Chain - Research Repository"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install runtime dependencies"
	@echo "  install-dev      Install all dependencies including dev tools"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run ruff linter"
	@echo "  format           Run black formatter"
	@echo "  type-check       Run mypy type checker"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run full test suite"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests"
	@echo "  test-regression  Run regression tests against paper values"
	@echo ""
	@echo "Research:"
	@echo "  reproduce        Full paper reproduction pipeline"
	@echo "  generate-data    Generate synthetic dataset"
	@echo "  run-simulation   Run federated learning simulation"
	@echo "  run-evaluation   Run evaluation pipeline"
	@echo "  generate-figures Generate paper figures"
	@echo "  generate-tables  Generate paper tables"
	@echo ""
	@echo "Infrastructure:"
	@echo "  docs             Build documentation"
	@echo "  docker-build     Build Docker images"
	@echo "  docker-up        Start all Docker services"
	@echo "  docker-down      Stop all Docker services"
	@echo "  clean            Remove generated files"

# Setup
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

# Code Quality
lint:
	ruff check src/ tests/ scripts/
	bandit -r src/ -ll

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

format-check:
	black --check src/ tests/ scripts/
	isort --check src/ tests/ scripts/

type-check:
	mypy src/

quality: lint format-check type-check

# Testing
test:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-regression:
	pytest tests/regression/ -v -m regression --timeout=600

test-fast:
	pytest tests/unit/ -v -x --tb=short

# Research Reproduction
generate-data:
	python scripts/generate_synthetic_data.py \
		--config configs/experiment/fedagent_chain_full.yaml \
		--seed 42 \
		--output-dir data/synthetic/

run-simulation:
	python scripts/run_federated_simulation.py \
		--config configs/experiment/fedagent_chain_full.yaml \
		--seed 42

run-baselines:
	python scripts/run_baselines.py \
		--config configs/experiment/baseline_local.yaml --seed 42
	python scripts/run_baselines.py \
		--config configs/experiment/baseline_centralized.yaml --seed 42

run-ablations:
	python scripts/run_ablation_study.py \
		--ablation-configs configs/experiment/ablation/ --seed 42

run-evaluation:
	python scripts/run_evaluation.py \
		--runs-dir experiments/runs/ \
		--results-dir experiments/results/ \
		--data-dir data/synthetic --seed 42 --seed-subdir

generate-figures:
	python scripts/generate_figures.py \
		--results-dir experiments/results/ \
		--runs-dir experiments/runs/

generate-tables:
	python scripts/generate_tables.py \
		--results-dir experiments/results/

export-audit:
	python scripts/export_blockchain_audit.py \
		--output-dir experiments/results/

reproduce: generate-data run-simulation run-baselines run-ablations \
           run-evaluation generate-figures generate-tables export-audit
	@echo "✅ Full paper reproduction complete. Check experiments/results/ and experiments/figures/"

# Documentation
docs:
	mkdocs build --clean

docs-serve:
	mkdocs serve --dev-addr 0.0.0.0:8080

docs-deploy:
	mkdocs gh-deploy --force --clean

# Docker
docker-build:
	docker build -f docker/Dockerfile -t fedagent-chain:latest .
	docker build -f docker/Dockerfile.gpu -t fedagent-chain:gpu .

docker-up:
	docker-compose -f docker/docker-compose.yml up --build -d
	@echo "Services starting... MLflow at http://localhost:5000, Dashboard at http://localhost:8501"

docker-down:
	docker-compose -f docker/docker-compose.yml down -v

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

# Dashboard
dashboard:
	streamlit run src/visualization/inclusion_dashboard.py --server.port 8501

# MLflow
mlflow-ui:
	mlflow server --host 0.0.0.0 --port 5000

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	rm -rf experiments/runs/* outputs/ multirun/
	@echo "Cleaned build artifacts"

clean-data:
	rm -rf data/synthetic/*.csv data/synthetic/*.parquet data/processed/
	@echo "Cleaned generated data"
