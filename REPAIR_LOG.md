# FedAgent-Chain Repair Log

## Repair Start Date: 2026-05-14
## Starting Commit: e0d6946
## Auditor Notes

### Critical Issues (Blockers)
1. [CI-1] Core fairness claim inverted: D_fair(disability) +105.9%, D_fair(work_mode) +3400%. The system is significantly increasing disparity while claiming to reduce it.
2. [CI-2] FedAvg delta accumulation divergence: loss 0.70→23.63 over 20 rounds. (Observed in paper report, to be verified in full runs).
3. [CI-3] Degenerate statistical tests: t=-inf, p=0, Cohen's d=-38e9 in per-seed CSVs. Indicates division by zero or zero-variance in metrics, likely due to identical results across seeds or flawed aggregation.
4. [CI-4] Local Baseline == Centralized (centralized flag never handled in data loader). Results are identical across these modes in committed artifacts.
5. [CI-5] _get_batch_group_labels uses wrong indices (fairness penalty on mismatched groups). Code audit required to verify indexing logic.

### Moderate Issues (Mandatory Fixes)
1. [MI-1] Missing random seed initialization in several components (e.g., `employment_model.py`).
2. [MI-2] Inconsistent F1 reporting between `results_temp/` (0.6734) and final paper (0.6374).
3. [MI-3] `Black` and `Safety` package versions in `requirements-dev.txt` have a dependency conflict on `packaging`.
4. [MI-4] `pyproject.toml` uses a non-standard build-backend (`setuptools.backends.legacy:build`) which fails on modern environments.
5. [MI-5] Data generation script doesn't enforce consistent user/job IDs across seeds unless explicitly set.

### Environment
- OS: Windows
- Python: 3.11.9
- PyTorch: 2.1.0+cpu
- CUDA available: No
- CPU cores: 12
- RAM: 15 GB

### Artifacts to Remove in Phase 7
- `debug_eval.py` — development debug file with hardcoded timestamps
- `per_round_debug.json` — unnamed provenance
- `experiments/results_temp/` — temp directory with inconsistent metrics
- Per-seed `table_2_multi_seed_summary.csv` files showing `F1_std=0.0`
