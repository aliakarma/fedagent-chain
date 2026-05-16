# 🛠️ Reproducibility Checklist

To verify the results presented in the FedAgent-Chain paper, follow these steps:

## 1. Environment Setup
- [ ] Create a clean Python 3.10+ environment.
- [ ] Install dependencies: `pip install -r requirements.txt`.
- [ ] Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`.

## 2. Data Generation
- [ ] Generate the synthetic dataset:
  ```bash
  python scripts/generate_synthetic_data.py --seed 42
  ```

## 3. Full Pipeline Execution
- [ ] Run the federated simulation for all 5 seeds (42, 123, 2024, 777, 999):
  ```bash
  python scripts/run_multi_seed_experiments.py
  ```
- [ ] Run the ablation study:
  ```bash
  python scripts/run_ablation_study.py
  ```

## 4. Evaluation & Results
- [ ] Aggregate results and generate tables:
  ```bash
  python scripts/aggregate_multi_seed_results.py
  ```
- [ ] Generate publication figures:
  ```bash
  python scripts/generate_figures.py
  ```
- [ ] Generate systems profiling plots:
  ```bash
  python scripts/generate_system_overhead_plots.py
  ```

## 5. Verification
- [ ] Compare the generated `experiments/results/table_2_classification_results.csv` against the values reported in the README.
- [ ] Verify blockchain audit integrity:
  ```bash
  python scripts/run_evaluation.py --seed 42
  ```
  (Look for "Chain Integrity: Valid" in the output).
