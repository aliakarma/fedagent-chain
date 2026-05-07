# Experiment Results

This directory contains committed CSV result files matching every table in the paper.

## Contents

| File | Paper Table | Description |
|------|-------------|-------------|
| `table_2_model_performance.csv` | Table 2 | Accuracy, Precision, Recall, F1, P@K, R@K |
| `table_3_fairness_results.csv` | Table 3 | Fairness disparity D_fair across protected groups |
| `table_4_blockchain_results.csv` | Table 4 | Blockchain auditability metrics |
| `table_5_agent_results.csv` | Table 5 | Agentic AI service evaluation |
| `table_6_accessibility.csv` | Table 6 | Accessibility outcome coverage |
| `table_7_overhead.csv` | Table 7 | System overhead analysis |

## Generating Results

```bash
# Generate synthetic data first
python scripts/generate_synthetic_data.py --seed 42

# Run full simulation
python scripts/run_federated_simulation.py \
    --config configs/experiment/fedagent_chain_full.yaml --seed 42

# Generate all tables and figures
python scripts/run_evaluation.py
python scripts/generate_figures.py
python scripts/generate_tables.py
```

Or use the single-command reproduction:
```bash
make reproduce
```

## Verifying Results

Run the regression test suite to verify CSV values match paper-reported numbers:
```bash
pytest tests/regression/ -v --timeout=600
```

Tolerance: ±0.005 for all reported metrics.
