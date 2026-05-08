# FedAgent-Chain: Secure Federated and Agentic AI for Multilingual Disability-Inclusive Employment

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org)
[![Flower Federated](https://img.shields.io/badge/FL-Flower-ff69b4.svg)](https://flower.dev)
[![Build Status](https://github.com/fedagent-chain/fedagent-chain/workflows/CI/badge.svg)](https://github.com/fedagent-chain/fedagent-chain/actions)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://fedagent-chain.github.io/fedagent-chain)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Official implementation of the paper:**
> *FedAgent-Chain: A Secure Federated and Agentic AI Framework for Multilingual Disability-Inclusive Employment in AI Cities*
> Toqeer Ali Syed, Muhammad Shoaib Siddiqui, et al. — *Frontiers in Artificial Intelligence*, 2025.

---

## Abstract

FedAgent-Chain is a secure federated and agentic AI framework designed to enable multilingual, disability-inclusive employment intelligence within AI-driven smart cities. The system unifies five advanced technology pillars — federated learning, blockchain-based auditability, agentic AI coordination, multilingual natural language processing, and human-in-the-loop governance — into a single cohesive architecture. The framework enables distributed institutions such as public employment agencies, universities, rehabilitation centers, employers, and assistive technology providers to collaboratively train inclusive employment models without ever centralizing sensitive disability-related data.

---

## Key Contributions

- **FedAgent-Chain Framework**: The first unified architecture combining federated learning, permissioned blockchain auditability, and agentic AI for disability-inclusive employment across multilingual, multi-institutional, cross-country settings.
- **Fairness-Aware Federated Learning**: A formalized fairness penalty integrated into the federated optimization objective, reducing disparity across disability categories, language groups, genders, and regional nodes.
- **Permissioned Blockchain Audit Layer**: Consent traceability, cryptographic model-update hashing, and smart-contract-based access control — without storing any raw disability data on-chain.
- **Five Specialized Agentic AI Services**: Employment matching, adaptive upskilling, workplace accommodation recommendation, multilingual communication, and human-in-the-loop governance agents.
- **Prototype Simulation**: A four-node cross-country simulation using synthetic disability-employment data demonstrating feasibility in matching performance, fairness, auditability, and system overhead.

---

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         Layer 1: Users & Stakeholders                       │
│  Persons with Disabilities | Employers | Advisors           │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│         Layer 3: Institutional Nodes (4 Countries)          │
│  Saudi Arabia | United States | China | Europe              │
│  Local Training | Data Preprocessing | Consent Mgmt         │
└──────────────────┬──────────────────────────────────────────┘
                   │ Protected Model Updates
┌──────────────────▼──────────────────────────────────────────┐
│         Layer 5: Federated Aggregation                      │
│  FedAvg | Fairness-Aware Optimizer | DP Noise               │
└──────────┬───────────────────────────────────────────────────┘
           │                          │ Audit Hashes
┌──────────▼──────────┐   ┌──────────▼──────────────────────┐
│  Layer 7: Agents    │   │  Layer 6: Blockchain             │
│  Employment Agent   │   │  Permissioned Chain              │
│  Upskilling Agent   │   │  Smart Contracts                 │
│  Accommodation      │   │  Consent Manager                 │
│  Multilingual       │   │  Audit Logger                    │
│  Governance Agent   │   └──────────────────────────────────┘
└─────────────────────┘
```

---

## Repository Structure

```
fedagent-chain/
├── configs/               # All experiment and system configurations (Hydra)
├── src/                   # Core framework source code
│   ├── federated/         # Federated learning (FedAvg, fairness, privacy)
│   ├── models/            # ML model definitions
│   ├── agents/            # Agentic AI service layer (5 agents)
│   ├── blockchain/        # Permissioned blockchain simulation
│   ├── data/              # Data loading, preprocessing, schema
│   ├── evaluation/        # Metrics, fairness, blockchain audit
│   ├── visualization/     # Plots and dashboards
│   └── utils/             # Shared utilities
├── nodes/                 # Per-country node simulation environments
├── data/                  # Schema templates and external dataset info
├── experiments/           # Results, figures, experiment logs
├── scripts/               # Entry-point scripts for training and evaluation
├── notebooks/             # Jupyter notebooks for demos and exploration
├── tests/                 # Unit, integration, and regression tests
├── docker/                # Docker and Docker Compose configuration
└── docs/                  # Extended documentation (MkDocs)
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- Docker 24.0+ and Docker Compose 2.20+ (for containerized deployment)

### Option A: Conda Environment (Recommended)

```bash
git clone https://github.com/aliakarma/fedagent-chain.git
cd fedagent-chain
conda env create -f environment.yml
conda activate fedagent-chain
pip install -e .
```

### Option B: pip with Virtual Environment

```bash
git clone https://github.com/aliakarma/fedagent-chain.git
cd fedagent-chain
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Option C: Docker (Fully Containerized)

```bash
git clone https://github.com/aliakarma/fedagent-chain.git
cd fedagent-chain
docker-compose -f docker/docker-compose.yml up --build
```

### Verify Installation

```bash
python -c "import src; print('FedAgent-Chain installed successfully')"
pytest tests/unit/ -v --tb=short
```

---

## Dataset Setup

### Generate Synthetic Dataset (Default)

```bash
python scripts/generate_synthetic_data.py \
    --config configs/experiment/fedagent_chain_full.yaml \
    --seed 42 \
    --output-dir data/synthetic/
```

This generates ~10,000 user profiles, 5,000 job profiles, and 50,000 suitability pairs across four regional nodes. See `data/synthetic/README.md` for the full schema.

### Download Public Datasets

```bash
python scripts/download_external_datasets.py --dataset onet
python scripts/download_external_datasets.py --dataset esco
```

---

## Reproducing Paper Results

### Full Framework (Tables 2–7, Figures 3–5)

```bash
# Step 1: Generate synthetic data
python scripts/generate_synthetic_data.py \
    --config configs/experiment/fedagent_chain_full.yaml --seed 42

# Step 2: Run federated simulation
python scripts/run_federated_simulation.py \
    --config configs/experiment/fedagent_chain_full.yaml --seed 42

# Step 3: Run evaluation
python scripts/run_evaluation.py \
    --config configs/evaluation/metrics.yaml \
    --results-dir experiments/results/

# Step 4: Generate figures and tables
python scripts/generate_figures.py --results-dir experiments/results/
python scripts/generate_tables.py  --results-dir experiments/results/
```

### Baselines

```bash
python scripts/run_baselines.py --config configs/experiment/baseline_local.yaml --seed 42
python scripts/run_baselines.py --config configs/experiment/baseline_centralized.yaml --seed 42
```

### Ablation Studies

```bash
python scripts/run_ablation_study.py \
    --ablation-configs configs/experiment/ablation/ --seed 42
```

### One-Command Full Reproduction

```bash
make reproduce  # Runs all steps above sequentially
```

---

## Evaluation Metrics

| Category | Metrics |
|---|---|
| Employment Matching | Accuracy, Precision, Recall, F1, P@K, R@K |
| Fairness | D_fair across disability, language, gender, region |
| Blockchain | Hash completeness, consent rate, transaction latency |
| Agentic AI | Accommodation relevance, upskilling quality, language adequacy |
| System Overhead | Round duration, memory usage, communication cost |

---

## Configuration Management

Experiments are managed via [Hydra](https://hydra.cc/). Override any parameter at runtime:

```bash
python scripts/run_federated_simulation.py \
    +experiment=fedagent_chain_full \
    federated.n_rounds=30 \
    privacy.noise_multiplier=0.5 \
    fairness.lambda_fairness=0.1 \
    seed=123
```

---

## Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Regression tests (verify paper results)
pytest tests/regression/ -v --timeout=600

# Full test suite with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Docker Usage

```bash
# Start all services (nodes, aggregator, blockchain, MLflow, dashboard)
docker-compose -f docker/docker-compose.yml up --build

# Run only the federated simulation
docker-compose -f docker/docker-compose.yml run fl-aggregator

# Access MLflow UI
open http://localhost:5000

# Access Streamlit dashboard
open http://localhost:8501
```

---

## Experiment Tracking

All runs are logged to MLflow:

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# View experiments in browser
open http://localhost:5000
```

Optional Weights & Biases integration:

```bash
export WANDB_API_KEY=your-key
python scripts/run_federated_simulation.py --config configs/experiment/fedagent_chain_full.yaml --use-wandb
```

---

## Citation

```bibtex
@article{syed2025fedagentchain,
  title     = {FedAgent-Chain: A Secure Federated and Agentic AI Framework for Multilingual Disability-Inclusive Employment in AI Cities},
  author    = {Syed, Toqeer Ali and Siddiqui, Muhammad Shoaib and Ali Akarma},
  journal   = {},
  year      = {2026},
  doi       = {10.xxxx/xxxxx}
}
```

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

The synthetic dataset (`data/synthetic/`) is released under **CC0 1.0 Universal** (public domain).

---

## Ethical Considerations

This research involves framework design and simulation for disability-inclusive AI systems. The prototype uses only synthetic data. **Any future real-world deployment requires:**

- Ethics review board (IRB/REC) approval
- Informed consent from all participants
- Compliance with applicable disability rights and data protection legislation (GDPR, ADA, PDPL, etc.)
- Human oversight for all employment-affecting decisions

See [docs/ethics.md](docs/ethics.md) for the complete ethical considerations statement.

The system must **never** be used to automatically reject a person with a disability's employment application without human review.

---

## Troubleshooting

**CUDA out of memory**: Reduce `model.batch_size` in `configs/default.yaml`.

**Flower connection refused**: Ensure the aggregator service is running before launching client nodes.

**Hash mismatch in blockchain tests**: Verify Python version consistency (3.10+) and UTF-8 encoding.

**Reproducibility issues**: Ensure `PYTHONHASHSEED=42` is set and `torch.backends.cudnn.deterministic=True`.

See [docs/troubleshooting.md](docs/troubleshooting.md) for the full troubleshooting guide.

---

## Acknowledgements

This research was supported by institutional grants and computing resources provided by the affiliated research institutions. The authors thank the open-source communities behind PyTorch, Flower, HuggingFace Transformers, and Hydra.

---

## Contact

For questions and issues, please open a [GitHub Issue](https://github.com/fedagent-chain/fedagent-chain/issues).

Corresponding author: [t.syed@institution.edu](mailto:t.syed@institution.edu)
