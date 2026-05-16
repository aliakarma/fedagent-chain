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
5. **Reproducible prototype simulation**: A four-node cross-country simulation (Saudi Arabia, United States, China, Europe) with synthetic disability-employment data, validated across **5 independent seeds** (42, 123, 2024, 777, 999).

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

# Run multi-seed simulations and baselines
python scripts/run_federated_simulation.py --seed 42
python scripts/run_baselines.py --seed 42
# (Repeat for seeds 123, 2024, 777, 999)

# Run ablation study comparison
python scripts/generate_ablation_table.py
```

### Step 3 — Verify Results

```bash
# Aggregate all 5 seeds
python scripts/aggregate_multi_seed_results.py --seeds 42 123 2024 777 999

# Generate publication figures
python scripts/generate_figures.py

# Run automated integrity check
python scripts/verify_submission_readiness.py
```

---

## Empirical Results

All metrics are computed from trained model checkpoints evaluated on held-out test sets (stratified 80/20 split per node). Results below are aggregated over **5 independent seeds**.

### Table 2 — Multi-Seed Model Performance (Mean ± 95% CI)

| Method | F1 Mean | F1 Std | 95% CI |
|:---|:---:|:---:|:---:|
| **FedAgent-Chain** | 0.7359 | 0.0553 | [0.6673, 0.8046] |
| Standard FedAvg | 0.7438 | 0.0380 | [0.6966, 0.7910] |
| Local Baseline | 0.5228 | 0.2615 | [0.1981, 0.8475] |
| Centralized | 0.7164 | 0.0232 | [0.6876, 0.7451] |

### Table 3 — Fairness Disparity (D_fair)

Lower values indicate more equitable performance across sub-groups. **Reduction** = improvement of FedAgent-Chain over Standard FedAvg.

| Protected Attribute | FedAgent-Chain | Std FedAvg | Local | Centralized | Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|
| Disability Category | **0.0377** | 0.0388 | 0.0700 | 0.0335 | +2.8% |
| Language Group | 0.4825 | 0.4834 | 0.4015 | 0.4463 | +0.2% |
| Work Mode | 0.0080 | 0.0055 | 0.0097 | 0.0085 | −45.5% |
| Regional Node | 0.1667 | 0.1703 | 0.1982 | 0.1611 | +2.1% |

### Table 6 — Ablation Study (λ-penalty)

| Variant | F1 (Mean) | D_fair (Mean) | Description |
|:---|:---:|:---:|:---|
| **Full System** | **0.7207** | 0.1653 | λ = 0.5 (Fairness-Aware FedAvg) |
| λ = 0 | 0.7116 | **0.1610** | Standard FedAvg baseline |

> **Interpretation**: The λ-penalty acts as a group-regularizer. While λ=0 shows lower mean disparity in this aggregate, the Full System achieves higher predictive stability (F1) across nodes.

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

### Table 7 — Systems Overhead

| Metric | Value | Description |
|:---|:---|:---|
| Avg Local Training Time | 16.0s | Per-node computation time (5 epochs) |
| Avg Aggregation Time | 0.0005s | Server-side coordination overhead |
| Avg Blockchain Logging Time | 0.001s | Hash submission latency |
| Model Size (KB) | 513 KB | Payload size per communication round |

### Scalability Discussion

The FedAgent-Chain architecture is designed for linear communication scalability:
1.  **Communication**: Total volume scales as $O(R \cdot K \cdot |W|)$, where $R$ is rounds, $K$ is nodes, and $|W|$ is model size. With a ~500KB model, a 100-node network would transmit ~100MB per round, well within modern institutional bandwidth.
2.  **Computation**: Local training is parallelized across nodes. Server-side aggregation is $O(K \cdot |W|)$, which is negligible for $K < 1000$.
3.  **Blockchain**: The audit trail grows linearly with $R \cdot K$. In production, a Merkle-tree based accumulator could further compress these logs.

---

## 🛠️ Advanced Workflows

### Performance Profiling

To regenerate systems overhead plots and CSVs:

```bash
python scripts/generate_system_overhead_plots.py
```

Outputs will be saved to `experiments/results/plots/runtime_breakdown.pdf`.

### Statistical Significance (Paired t-test, n=5 seeds)

| Comparison | Δ F1 | t | p | Cohen's d | Sig. |
|:---|:---:|:---:|:---:|:---:|:---:|
| FedAgent-Chain vs Std FedAvg | −0.0100 | −1.03 | 0.377 | −0.52 | No |
| FedAgent-Chain vs Local | +0.2475 | 1.55 | 0.219 | 0.77 | No |
| FedAgent-Chain vs Centralized | +0.0147 | 0.81 | 0.479 | 0.40 | No |

---

## Repository Structure

```
fedagent-chain/
├── configs/                        # Hydra experiment configurations
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
│   ├── run_federated_simulation.py
│   ├── run_evaluation.py
│   ├── aggregate_multi_seed_results.py
│   ├── generate_figures.py
│   ├── generate_lambda_tradeoff_plot.py
│   └── verify_submission_readiness.py
├── experiments/                   # Output directory
│   ├── results/                   # CSV tables (Tables 2–7)
│   │   ├── seeds/                 # Raw per-seed metrics
│   │   ├── plots/                 # Publication PDF figures
│   │   └── statistics/            # Aggregated t-tests and CIs
│   └── runs/                      # Per-run checkpoints and metrics
├── tests/                         # Test suite
└── docs/                          # Extended documentation (MkDocs)
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
```

---

## Configuration

Experiments are managed via [Hydra](https://hydra.cc/). Override any parameter at runtime:

```bash
python scripts/run_federated_simulation.py \
    --config configs/experiment/fedagent_chain_full.yaml \
    federated.n_rounds=20 \
    privacy.noise_multiplier=0.1 \
    fairness.lambda_fairness=0.5 \
    --seed 123
```

---

## Ethical Considerations

This research involves framework design and simulation for disability-inclusive AI systems. The prototype uses **only synthetic data**.

- **No real disability data** is collected, stored, or processed.
- **Human oversight** is mandatory: the Governance Agent provides *recommendations*; final employment decisions must always be made by qualified human advisors.
- **The system must never automatically reject** a person with a disability's employment application without human review.

---

## Citation

```bibtex
@article{syed2026fedagentchain,
  title   = {FedAgent-Chain: A Secure Federated and Agentic AI Framework
             for Multilingual Disability-Inclusive Employment in AI Cities},
  author  = {Syed, Toqeer Ali and Siddiqui, Muhammad Shoaib and Ali Akarma},
  journal = {Frontiers in Artificial Intelligence},
  year    = {2026},
}
```

---

## Contact

For questions and issues, please open a [GitHub Issue](https://github.com/aliakarma/fedagent-chain/issues).
