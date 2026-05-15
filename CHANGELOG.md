# Changelog

All notable changes to FedAgent-Chain are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.1.0-repaired] - 2026-05-15

### Added
- **REPAIR_LOG.md**: Comprehensive documentation of the remediation process.
- **Statistical Guards**: 4-point verification in `aggregate_multi_seed_results.py` to prevent degenerate statistics.
- **Artifact Integrity**: `scripts/verify_artifact_integrity.py` and `scripts/validate_checkpoints.py` for repository hardening.
- **CI Hardening**: Automated loss divergence detection in GitHub Actions.
- **Fairness Note**: README update with honest statistical interpretation disclaimer.

### Fixed
- **Fairness Inversion [CI-1]**: Corrected weight formula to incentivize fairness instead of maximizing disparity.
- **Mode Collapse [CI-4]**: Fixed centralized baseline pooling to correctly combine regional datasets.
- **Numerical Stability**: Replaced delta-accumulation with absolute weight averaging; added `LayerNorm` for FL stability.
- **Fairness Labeling [CI-5]**: Fixed batch-slicing mismatch in protected group label extraction.
- **Evaluation Logic**: Fixed dead code in checkpoint selection and restored fallback reachability.

### Removed
- **Debug Artifacts**: Removed `debug_eval.py`, `per_round_debug.json`, and `experiments/results_temp/`.

---

## [1.0.0] — 2025-01-01

### Added — Initial public release accompanying paper submission

**Core Framework**
- Complete FedAgent-Chain federated learning framework across four regional nodes
  (Saudi Arabia, United States, China, Europe)
- Standard FedAvg aggregator (`src/federated/aggregator.py`)
- Fairness-Aware FedAvg aggregator with per-node fairness adjustment factor ρ_k
- Differential privacy: L2 gradient clipping + Gaussian noise (`src/federated/privacy.py`)
- Fairness regularization objective Ω_fair and disparity metric D_fair (`src/federated/fairness.py`)
- Federated client with local training and DP-protected update submission (`src/federated/client.py`)
- Federated server orchestrating multi-round training with MLflow logging (`src/federated/server.py`)

**Blockchain Audit Layer**
- Permissioned blockchain simulation with SHA-256 cryptographic hashing
- Model update hash submission: h_k^t = H(Δw̃_k || ID_k || t)
- Governance event recording for human-in-the-loop oversight
- Chain integrity verification and JSON audit log export

**Agentic AI Services**
- `BaseAgent` abstract class with governance triggering and structured audit logging
- `EmploymentAgent`: suitability scoring S(i,r) = α·sim + β·A + γ·L + δ·P

**Data Layer**
- Pydantic-validated schema for UserProfile, JobProfile, EmploymentOutcome, BlockchainRecord
- Synthetic disability-employment data generator (10K users, 5K jobs, 50K pairs across 4 nodes)
- PyTorch Dataset with consent filtering and fairness reweighting

**Evaluation**
- Full classification metrics: Accuracy, Precision, Recall, F1
- Ranking metrics: Precision@K and Recall@K (K = 5, 10)
- Fairness disparity evaluator across all protected attributes
- Regression test suite validating paper-reported values (tolerance ±0.005)

**Infrastructure**
- Hydra-based configuration management with full experiment YAML suite
- MLflow experiment tracking integration
- Docker + Docker Compose multi-node simulation environment
- GitHub Actions CI/CD (lint, unit, integration, regression, security, smoke tests)
- Pre-commit hooks (ruff, black, isort, mypy, bandit)
- MkDocs documentation site with mkdocstrings API reference

**Scripts**
- `generate_synthetic_data.py` — reproducible dataset generation
- `run_federated_simulation.py` — main simulation entry point
- `run_evaluation.py` — full paper table generation
- `generate_figures.py` — paper figures 3, 4, 5
- `run_ablation_study.py` — all ablation experiments
- `export_blockchain_audit.py` — audit log export

---

## Upcoming

### [1.1.0] — Planned

- [ ] FedProx aggregator (heterogeneous data robustness)
- [ ] SCAFFOLD client-drift correction
- [ ] Formal (ε, δ)-DP budget tracker using Opacus
- [ ] Hyperledger Fabric integration replacing in-memory blockchain
- [ ] Upskilling, Accommodation, Multilingual, and Governance agent full implementations
- [ ] Kubernetes Helm chart for distributed deployment
- [ ] PapersWithCode benchmark submission
