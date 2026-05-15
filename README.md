# FedAgent-Chain

**A Secure Federated and Agentic AI Framework for Multilingual Disability-Inclusive Employment in AI Cities**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Syed, Toqeer Ali · Siddiqui, Muhammad Shoaib · Ali Akarma
> *Frontiers in Artificial Intelligence*, 2026

---

## Table of Contents

- [Abstract](#abstract)
- [Key Contributions](#key-contributions)
- [Framework Architecture](#framework-architecture)
- [Quick Start — Reproduce in 3 Steps](#quick-start--reproduce-in-3-steps)
- [Empirical Results](#empirical-results)
- [Step-by-Step Commands](#step-by-step-commands)
- [Repository Structure](#repository-structure)
- [Testing](#testing)
- [Configuration](#configuration)
- [Ethical Considerations](#ethical-considerations)
- [Citation](#citation)
- [License](#license)

---

## Abstract

FedAgent-Chain is a unified framework that enables distributed institutions — public employment agencies, universities, rehabilitation centres, employers, and assistive-technology providers — to **collaboratively train inclusive employment models without ever centralising sensitive disability data**. The system integrates five technology pillars:

| Pillar | Purpose |
|:---|:---|
| **Federated Learning** | Privacy-preserving distributed model training across 4 regional nodes |
| **Fairness-Aware Aggregation** | Novel λ-penalty in FedAvg that reduces disparity across protected groups |
| **Permissioned Blockchain** | Immutable audit trail for model updates and consent management |
| **Agentic AI Services** | 5 specialised agents for matching, upskilling, accommodation, multilingual support, and governance |
| **Differential Privacy** | Gradient clipping and calibrated noise injection for formal privacy guarantees |

---

## Key Contributions

1. **First unified architecture** combining federated learning, permissioned blockchain, and agentic AI for disability-inclusive employment across multilingual, multi-institutional, cross-country settings.
2. **Fairness-Aware FedAvg**: A formalised fairness penalty (λ) integrated into the federated optimisation objective, with an empirical Pareto frontier characterising the accuracy–fairness tradeoff.
3. **Permissioned blockchain audit layer**: Consent traceability, cryptographic model-update hashing, and smart-contract-based access control — without storing raw disability data on-chain.
4. **Five specialised agentic AI services**: Employment matching, adaptive upskilling, workplace accommodation, multilingual communication, and human-in-the-loop governance.
5. **Reproducible prototype simulation**: A four-node cross-country simulation (Saudi Arabia, United States, China, Europe) with synthetic disability-employment data, validated across 3 independent seeds.

---

## Framework Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1 — Users & Stakeholders                              │
│  Persons with Disabilities · Employers · Vocational Advisors │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  Layer 2 — Data Ingestion                                    │
│  Synthetic Dataset · O*NET · ESCO · Regional Disability Data │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  Layer 3 — Institutional Nodes (4 Countries)                 │
│  Saudi Arabia │ United States │ China │ Europe               │
│  Local Training · Data Preprocessing · Consent Management    │
└────────────────────────┬─────────────────────────────────────┘
                         │ DP-Protected Model Updates
┌────────────────────────▼─────────────────────────────────────┐
│  Layer 4 — Security & Privacy                                │
│  Differential Privacy (ε,δ) · LayerNorm Stabilisation        │
│  Gradient Clipping (C=1.0) · Noise Multiplier (σ=0.1)       │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  Layer 5 — Federated Aggregation                             │
│  Standard FedAvg │ Fairness-Aware FedAvg (λ-penalty)        │
│  Weight formula: ρ_i = 1 + λ · min-group-F1_i               │
└─────────┬────────────────────────────────────────────────────┘
          │                              │ Audit Hashes
┌─────────▼───────────────┐  ┌───────────▼────────────────────┐
│  Layer 7 — Agentic AI   │  │  Layer 6 — Blockchain          │
│  Employment Matching    │  │  Permissioned Ledger           │
│  Upskilling Agent       │  │  SHA-256 Hash Chain            │
│  Accommodation Agent    │  │  Smart Contracts               │
│  Multilingual Agent     │  │  Consent Logger                │
│  Governance Agent       │  │  Audit Trail                   │
└─────────────────────────┘  └────────────────────────────────┘
```

---

## Quick Start — Reproduce in 3 Steps

> **For reviewers**: These 3 steps reproduce all paper tables and figures from scratch.
> Total runtime: ~15 minutes on a modern CPU.

### Step 1 — Install

```bash
git clone https://github.com/aliakarma/fedagent-chain.git
cd fedagent-chain
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Step 2 — Generate Data & Run All Experiments

```bash
python scripts/generate_synthetic_data.py \
    --config configs/experiment/fedagent_chain_full.yaml --seed 42

python scripts/reproduce_paper_results.py
```

### Step 3 — Verify Results

```bash
# Tables are in experiments/results/
cat experiments/results/table_2_model_performance.csv
cat experiments/results/table_3_fairness_results.csv

# Figures are in experiments/figures/
ls experiments/figures/*.pdf

# Run automated integrity check
python scripts/verify_artifact_integrity.py
```

---

## Empirical Results

All metrics are computed from trained model checkpoints evaluated on held-out test sets (stratified 80/20 split per node). Results below are from **Seed 42**; multi-seed statistics (n=3) are in `experiments/results/table_2_multi_seed_summary.csv`.

### Table 2 — Model Performance

| Method | Accuracy | Precision | Recall | F1 | P@5 | P@10 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **FedAgent-Chain** | 0.7534 | 0.7374 | 0.7996 | **0.7627** | 1.00 | 0.95 |
| Standard FedAvg | 0.7541 | 0.7395 | 0.7951 | 0.7621 | 1.00 | 0.95 |
| Centralized | 0.7327 | 0.7590 | 0.6960 | 0.7237 | 0.95 | 0.85 |
| Local Baseline | 0.5377 | 0.5331 | 0.9826 | 0.6868 | 0.75 | 0.68 |

**Multi-seed summary (mean ± std over seeds 42, 123, 2024):**

| Method | F1 (mean ± std) | 95% CI |
|:---|:---:|:---:|
| **FedAgent-Chain** | 0.7599 ± 0.0064 | [0.7439, 0.7759] |
| Standard FedAvg | 0.7602 ± 0.0059 | [0.7455, 0.7749] |
| Centralized | 0.7273 ± 0.0077 | [0.7082, 0.7465] |
| Local Baseline | 0.4170 ± 0.3078 | — |

### Table 3 — Fairness Disparity (D_fair)

Lower values indicate more equitable performance across sub-groups. **Reduction** = improvement of FedAgent-Chain over Standard FedAvg.

| Protected Attribute | FedAgent-Chain | Std FedAvg | Local | Centralized | Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|
| Disability Category | **0.0377** | 0.0388 | 0.0700 | 0.0335 | +2.8% |
| Language Group | 0.4825 | 0.4834 | 0.4015 | 0.4463 | +0.2% |
| Work Mode | 0.0080 | 0.0055 | 0.0097 | 0.0085 | −45.5% |
| Regional Node | 0.1667 | 0.1703 | 0.1982 | 0.1611 | +2.1% |

### Table 4 — Blockchain Audit

| Metric | Value | Description |
|:---|:---:|:---|
| Hash Completeness | **100%** | All model updates have valid SHA-256 hashes |
| Chain Integrity | **Valid** | Full hash-chain verification passed |
| Total Audit Records | 40 | 4 nodes × 10 rounds |
| Chain Length | 5 blocks | Finalised on-chain blocks |

### Table 5 — Agentic AI Services

| Agent | Metric | Score |
|:---|:---|:---:|
| Employment Matching | Mean Confidence | 0.694 |
| Upskilling | Skill Gap Coverage | **1.000** |
| Accommodation | Accommodation Coverage | 0.731 |
| Multilingual | Language Adequacy | **0.969** |
| Governance | High-Risk Detection Rate | 0.733 |
| Governance | False Positive Rate | 0.060 |

### Table 7 — System Overhead

| Component | Time | Notes |
|:---|:---:|:---|
| Avg round duration | 86.6 s | Mean across all FL rounds (4 nodes, 3 local epochs) |
| Min / Max round | 71.2 s / 108.4 s | — |
| Total simulation | 866.4 s | 10 rounds × 4 nodes |

### Statistical Significance (Paired t-test, n=3 seeds)

| Comparison | Δ F1 | t | p | Cohen's d | Sig. |
|:---|:---:|:---:|:---:|:---:|:---:|
| FedAgent-Chain vs Std FedAvg | −0.0004 | −0.38 | 0.744 | −0.22 | No |
| FedAgent-Chain vs Local | +0.3429 | 1.89 | 0.199 | 1.09 | No |
| FedAgent-Chain vs Centralized | +0.0325 | 6.25 | **0.025** | **3.61** | **Yes** |

> **Note:** With n=3 seeds, statistical power is limited. Effect sizes (Cohen's d) and 95% confidence intervals are provided in `experiments/results/table_2_multi_seed_summary.csv`. Non-significant comparisons should be interpreted as directional trends.

### Figure Summary

| Figure | File | Description |
|:---|:---|:---|
| FL Convergence | `experiments/figures/fl_convergence.pdf` | Training loss and F1 across rounds |
| Per-Node F1 | `experiments/figures/node_f1_scores.pdf` | F1 breakdown by regional node |
| Fairness Disparity | `experiments/figures/fairness_disparity.pdf` | D_fair comparison across methods |
| λ Tradeoff | `experiments/figures/lambda_tradeoff.pdf` | Pareto frontier: F1 vs D_fair for λ ∈ {0, 0.05, …, 2.0} |

---

## Step-by-Step Commands

For reviewers who prefer to run each experiment individually rather than using the one-click script.

### 1. Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py \
    --config configs/experiment/fedagent_chain_full.yaml \
    --seed 42 \
    --output-dir data/synthetic/
```

Generates ~10,000 user profiles, ~5,000 job profiles, and ~50,000 suitability pairs across 4 regional nodes.

### 2. Run FedAgent-Chain (Main Method)

```bash
python scripts/run_federated_simulation.py \
    --config configs/experiment/fedagent_chain_full.yaml \
    --seed 42 --no-mlflow
```

### 3. Run Standard FedAvg Ablation

```bash
python scripts/run_federated_simulation.py \
    --config configs/experiment/ablation/no_fairness.yaml \
    --seed 42 --no-mlflow
```

### 4. Run Baselines

```bash
# Local Baseline (each node trains independently)
python scripts/run_federated_simulation.py \
    --config configs/experiment/baseline_local.yaml \
    --seed 42 --no-mlflow

# Centralized Baseline (all data pooled)
python scripts/run_federated_simulation.py \
    --config configs/experiment/baseline_centralized.yaml \
    --seed 42 --no-mlflow
```

### 5. Evaluate & Generate Tables

```bash
python scripts/run_evaluation.py \
    --runs-dir experiments/runs/ \
    --results-dir experiments/results/ \
    --data-dir data/synthetic \
    --seed 42
```

### 6. Multi-Seed Aggregation

Repeat Steps 2–5 with `--seed 123` and `--seed 2024`, then aggregate:

```bash
python scripts/aggregate_multi_seed_results.py \
    --seeds 42 123 2024 \
    --results-dir experiments/results/
```

### 7. Lambda Fairness Sweep (Figure 6)

```bash
python scripts/run_lambda_sweep.py
python scripts/generate_lambda_tradeoff_plot.py
```

### 8. Generate Publication Figures

```bash
python scripts/generate_figures.py \
    --results-dir experiments/results/ \
    --runs-dir experiments/runs/ \
    --output-dir experiments/figures/
```

### 9. Verify Submission Readiness

```bash
python scripts/verify_submission_readiness.py
python scripts/verify_artifact_integrity.py
python scripts/validate_checkpoints.py
```

---

## Repository Structure

```
fedagent-chain/
├── configs/                        # Hydra experiment configurations
│   ├── default.yaml                # Default hyperparameters
│   ├── experiment/                 # Per-experiment configs
│   │   ├── fedagent_chain_full.yaml
│   │   ├── baseline_local.yaml
│   │   ├── baseline_centralized.yaml
│   │   ├── ablation/              # Ablation studies
│   │   └── lambda_sweep/          # λ ∈ {0.00, 0.05, …, 2.00}
│   └── nodes/                     # Per-country node configs
├── src/                           # Core framework source code
│   ├── federated/                 # FedAvg, Fairness-Aware aggregator, DP
│   ├── models/                    # Neural network (MLP + LayerNorm)
│   ├── agents/                    # 5 specialised agentic services
│   ├── blockchain/                # Permissioned chain, smart contracts
│   ├── data/                      # Dataset loading, schema, preprocessing
│   ├── evaluation/                # Metrics, fairness computation, audit
│   ├── visualization/             # Plotting utilities
│   └── utils/                     # Helpers, logging, seeding
├── scripts/                       # Entry-point scripts
│   ├── reproduce_paper_results.py # One-click full reproduction
│   ├── run_federated_simulation.py
│   ├── run_evaluation.py
│   ├── run_lambda_sweep.py
│   ├── generate_figures.py
│   ├── aggregate_multi_seed_results.py
│   ├── verify_artifact_integrity.py
│   └── validate_checkpoints.py
├── experiments/                   # Output directory
│   ├── results/                   # CSV tables (Tables 2–7)
│   ├── figures/                   # PDF plots (Figures 3–6)
│   └── runs/                      # Per-run checkpoints and metrics
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── regression/                # Regression tests (paper result anchors)
├── data/                          # Dataset schemas and synthetic data
├── docker/                        # Docker and Docker Compose configs
├── docs/                          # Extended documentation (MkDocs)
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── REPAIR_LOG.md                  # Remediation audit trail
└── CHANGELOG.md                   # Version history
```

---

## Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v -m integration --timeout=120

# Regression tests (anchored to paper results)
pytest tests/regression/ -v -m regression --timeout=300

# Full suite with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Configuration

Experiments are managed via [Hydra](https://hydra.cc/). Override any parameter at runtime:

```bash
python scripts/run_federated_simulation.py \
    --config configs/experiment/fedagent_chain_full.yaml \
    federated.n_rounds=30 \
    privacy.noise_multiplier=0.5 \
    fairness.lambda_fairness=0.1 \
    --seed 123
```

Key hyperparameters:

| Parameter | Default | Description |
|:---|:---:|:---|
| `federated.n_rounds` | 10 | Number of FL communication rounds |
| `federated.local_epochs` | 3 | Local training epochs per round |
| `federated.learning_rate` | 0.001 | Client learning rate |
| `fairness.lambda_fairness` | 0.5 | Fairness penalty weight (λ) |
| `privacy.noise_multiplier` | 0.1 | DP noise scale (σ) |
| `privacy.clipping_threshold` | 1.0 | Gradient clipping bound (C) |

---

## Ethical Considerations

This research involves framework design and simulation for disability-inclusive AI systems. The prototype uses **only synthetic data**.

- **No real disability data** is collected, stored, or processed.
- **Human oversight** is mandatory: the Governance Agent provides *recommendations*; final employment decisions must always be made by qualified human advisors.
- **Any future real-world deployment requires**: IRB/ethics board approval, informed consent from all participants, and compliance with GDPR, ADA, PDPL, and applicable disability rights legislation.
- **The system must never automatically reject** a person with a disability's employment application without human review.

See [docs/ethics.md](docs/ethics.md) for the full ethical considerations statement.

---

## Troubleshooting

| Issue | Solution |
|:---|:---|
| CUDA out of memory | Reduce `federated.batch_size` in config |
| Reproducibility issues | Ensure `PYTHONHASHSEED=42` and `torch.backends.cudnn.deterministic=True` |
| Hash mismatch in blockchain tests | Verify Python 3.10+ and UTF-8 encoding |
| Import errors | Run `pip install -e .` and ensure `PYTHONPATH` includes project root |

---

## Citation

```bibtex
@article{syed2025fedagentchain,
  title   = {FedAgent-Chain: A Secure Federated and Agentic AI Framework
             for Multilingual Disability-Inclusive Employment in AI Cities},
  author  = {Syed, Toqeer Ali and Siddiqui, Muhammad Shoaib and Ali Akarma},
  journal = {Frontiers in Artificial Intelligence},
  year    = {2026},
  doi     = {10.xxxx/xxxxx}
}
```

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.
The synthetic dataset (`data/synthetic/`) is released under **CC0 1.0 Universal** (public domain).

---

## Contact

For questions and issues, please open a [GitHub Issue](https://github.com/aliakarma/fedagent-chain/issues).
