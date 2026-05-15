# MASTER REMEDIATION PLAN: FedAgent-Chain Repository Repair

## Context and Mandate

You are a **senior ML systems engineer, NeurIPS reproducibility reviewer, and scientific auditor** tasked with repairing the FedAgent-Chain research repository. A full peer review audit has been completed and has identified five critical rejection-level issues, five moderate issues, and four required missing experiments. Your job is to execute a systematic, phase-gated remediation plan. You must think simultaneously as:

- A **federated learning systems engineer** who understands gradient flow, DP noise accumulation, and FedAvg convergence properties
- A **fairness ML researcher** who understands protected attribute alignment, surrogate penalties, and disparity metrics
- A **scientific auditor** who verifies every number can be traced to runnable code
- A **production ML architect** who enforces reproducibility, determinism, and artifact provenance
- A **NeurIPS reproducibility reviewer** who will reject any result not supported by verifiable computation

**Repository under repair:** FedAgent-Chain (federated learning + blockchain + agentic AI for disability-inclusive employment)

**Primary source of truth:** The review report. Every issue flagged as CRITICAL is a hard blocker. Every MODERATE issue is a mandatory fix unless you provide written architectural justification for deferral. Every MISSING EXPERIMENT is required research validation.

**Non-negotiable rules throughout all phases:**
- NEVER claim a fix is complete without running the associated test suite and observing passing results
- NEVER skip an experiment to save time
- NEVER assume metrics improved — measure them
- NEVER fabricate statistical significance
- NEVER silently ignore failed tests or warnings
- NEVER overwrite committed artifacts without first creating a timestamped backup branch
- NEVER continue to the next phase when success criteria have not been met
- NEVER produce placeholder values in any CSV, JSON, or table
- ALL random seeds must be set before ANY stochastic operation
- ALL experiments must log hardware info, git commit hash, and config hash

---

## Phase 0 — Repository Audit, Backup, and Environment Verification

### Objective
Establish a clean, verified baseline state before any code modification. Create backup branches. Verify the development environment exactly matches the paper's stated dependencies. Document the current broken state with checksums so that any regression introduced during repair is immediately detectable.

### Why This Phase Matters
You cannot repair a repository you don't fully understand. Every artifact currently committed may be either correct, partially correct, or fabricated. You must establish ground truth before making changes. Modifying code without a backup means losing the ability to compare pre-fix vs post-fix behavior.

### Files Involved
All files in the repository. Priority audit targets:
- `experiments/results/**/*.csv` and `experiments/results/**/*.json`
- `experiments/verification_run/**`
- `experiments/results_temp/**`
- `debug_eval.py` (root level)
- `per_round_debug.json` (root level)
- `src/federated/server.py`
- `src/federated/client.py`
- `src/federated/aggregator.py`
- `src/models/employment_model.py`
- `scripts/run_evaluation.py`
- `scripts/run_federated_simulation.py`
- `scripts/aggregate_multi_seed_results.py`
- `configs/experiment/fedagent_chain_full.yaml`
- `configs/experiment/ablation/no_fairness.yaml`
- `configs/experiment/baseline_centralized.yaml`

### Tasks

**Task 0.1 — Create Backup Branch**
```bash
git checkout main
git pull origin main
git checkout -b pre-repair-backup-$(date +%Y%m%d-%H%M%S)
git push origin pre-repair-backup-$(date +%Y%m%d-%H%M%S)
git checkout main
git checkout -b repair/phase-0-audit
```
Document the commit hash you are starting from. This hash must appear in every subsequent commit message.

**Task 0.2 — Environment Verification**
Verify that your Python environment exactly matches `requirements.txt` and `environment.yml`:
```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
python -c "import torch; import sklearn; import flwr; import omegaconf; import pydantic; print('All imports OK')"
python -c "import torch; assert torch.__version__ == '2.1.0', f'Wrong PyTorch: {torch.__version__}'"
python -c "import sklearn; assert sklearn.__version__ == '1.3.2', f'Wrong sklearn: {sklearn.__version__}'"
```
If any version mismatches exist, document them. Do not proceed if core ML packages differ.

**Task 0.3 — Full Test Suite Baseline Run**
Run the existing test suite and record the current pass/fail state:
```bash
pytest tests/unit/ -v --tb=short 2>&1 | tee logs/phase0_unit_baseline.txt
pytest tests/integration/ -v --tb=short 2>&1 | tee logs/phase0_integration_baseline.txt
pytest tests/regression/ -v --tb=short 2>&1 | tee logs/phase0_regression_baseline.txt
```
Every failing test is a pre-existing failure. Record all of them. You must not introduce NEW failures; you may only fix existing ones.

**Task 0.4 — Artifact Checksum Registry**
Generate SHA-256 checksums for all committed result artifacts:
```bash
find experiments/results/ -name "*.csv" -o -name "*.json" | sort | xargs sha256sum > logs/phase0_artifact_checksums_before.txt
find experiments/ -name "*.pt" | sort | xargs sha256sum >> logs/phase0_artifact_checksums_before.txt
echo "Checksum registry created at $(date)" >> logs/phase0_artifact_checksums_before.txt
```
This file must be committed and will be used in Phase 8 to verify that regenerated results are fully self-consistent.

**Task 0.5 — Document Broken State**
Create `REPAIR_LOG.md` at the repository root with the following mandatory sections:
```markdown
# FedAgent-Chain Repair Log

## Repair Start Date: [DATE]
## Starting Commit: [HASH]
## Auditor Notes

### Critical Issues (Blockers)
1. [CI-1] Core fairness claim inverted: D_fair(disability) +106%, D_fair(work_mode) +3400%
2. [CI-2] FedAvg delta accumulation divergence: loss 0.70→23.63 over 20 rounds
3. [CI-3] Degenerate statistical tests: t=-inf, p=0, Cohen's d=-38e9 in per-seed CSVs
4. [CI-4] Local Baseline == Centralized (centralized flag never handled in data loader)
5. [CI-5] _get_batch_group_labels uses wrong indices (fairness penalty on mismatched groups)

### Moderate Issues (Mandatory Fixes)
...

### Environment
- OS: [OS]
- Python: [VERSION]
- PyTorch: [VERSION]
- CUDA available: [YES/NO]
- CPU cores: [N]
- RAM: [GB]
```

**Task 0.6 — Identify and Flag Dead Artifacts**
The following files have been identified as problematic artifacts that must be audited before deletion in Phase 7:
- `debug_eval.py` — development debug file with hardcoded timestamps
- `per_round_debug.json` — unnamed provenance
- `experiments/results_temp/` — temp directory with inconsistent metrics (FedAgent F1=0.6734 vs reported 0.6374)
- Per-seed `table_2_multi_seed_summary.csv` files (in seed_42/, seed_123/, seed_2024/) showing `F1_std=0.0` — invalid aggregation artifacts

Do NOT delete these yet. Mark them with a comment in `REPAIR_LOG.md` under "Artifacts to Remove in Phase 7".

**Task 0.7 — Smoke Test: Can the Pipeline Run at All?**
Run the minimal smoke test pipeline with 2 rounds to confirm the environment works:
```bash
python scripts/generate_synthetic_data.py \
    --config configs/experiment/fedagent_chain_full.yaml \
    --seed 42 \
    --n-users-per-node 100 \
    --n-jobs-per-node 50 \
    --n-pairs-per-node 200 2>&1 | tee logs/phase0_smoke_data.txt

python scripts/run_federated_simulation.py \
    --config configs/experiment/fedagent_chain_full.yaml \
    --seed 42 \
    --no-mlflow \
    --override federated.n_rounds=2 federated.local_epochs=1 \
    2>&1 | tee logs/phase0_smoke_simulation.txt
```
If the smoke test fails, stop immediately and diagnose the import/runtime error before proceeding.

### Architecture Considerations
None — this is an audit phase. No architectural changes are made.

### Validation
- All baseline test results documented
- Checksum registry committed
- REPAIR_LOG.md created and committed
- Smoke test passes
- Environment versions verified

### Expected Outputs
- `logs/phase0_unit_baseline.txt`
- `logs/phase0_integration_baseline.txt`
- `logs/phase0_regression_baseline.txt`
- `logs/phase0_artifact_checksums_before.txt`
- `logs/phase0_smoke_data.txt`
- `logs/phase0_smoke_simulation.txt`
- `REPAIR_LOG.md`

### Success Criteria
- [ ] Backup branch exists on remote
- [ ] Environment versions match requirements.txt within major.minor
- [ ] All pre-existing test results documented (pass/fail counts recorded)
- [ ] Artifact checksum registry created
- [ ] REPAIR_LOG.md committed with all 5 critical issues and 5 moderate issues listed
- [ ] Smoke test (2 rounds, 4 nodes, 100 users/node) completes without exception
- [ ] `logs/` directory committed with all baseline files

### Failure Criteria
- Environment package versions differ from requirements.txt for core ML dependencies AND you cannot establish an exact-match environment → STOP and document the version delta
- Smoke test raises an import error or crashes mid-simulation → STOP and fix import/runtime before proceeding
- Git backup branch does not exist on remote → STOP

### Stop Conditions
If the smoke test fails with a crash (not a numerical issue, but an actual runtime exception), do not proceed to Phase 1 until it is fixed. A crash indicates a broken baseline that would contaminate all downstream phases.

### Git Commit Recommendation
```
git add logs/ REPAIR_LOG.md
git commit -m "phase0: audit baseline, artifact checksums, repair log init [starting-commit: HASH]"
```

---

**PHASE COMPLETION CHECKLIST — PHASE 0**

**SUCCESS CONDITIONS:**
- Backup branch on remote ✓
- Environment verified ✓
- Baseline test results documented ✓
- Checksum registry committed ✓
- REPAIR_LOG.md committed ✓
- Smoke test passes ✓

**FAIL CONDITIONS:**
- Any import error in smoke test
- Core ML package version mismatch not documented
- Backup branch not pushed to remote

**NEXT PHASE ENTRY REQUIREMENTS:**
All success conditions must be checked. REPAIR_LOG.md must exist. Smoke test log must show no exception.

**DO NOT CONTINUE TO PHASE 1 UNTIL ALL SUCCESS CRITERIA PASS.**

---

## Phase 1 — Core Federated Learning Stability Fix (FedAvg Delta Accumulation Divergence)

### Objective
Fix the root cause of training loss divergence in the Standard FedAvg configuration. Currently `FederatedServer` applies accumulated DP-noisy deltas to the global state every round, causing unbounded weight growth (BCE loss: 0.70 → 23.63 over 20 rounds). Replace the delta-based update with proper FedAvg absolute-weight averaging. Verify that training loss decreases or stabilizes over 20 rounds.

### Why This Phase Matters
This is the most foundational fix in the entire repair plan. Every downstream comparison (FedAgent-Chain vs Standard FedAvg), every fairness measurement, and every statistical test depends on having a valid, non-diverging Standard FedAvg baseline. If the baseline diverges, then:
1. The "Standard FedAvg" F1 of 0.6762 (Table 2) comes from the best checkpoint at round 1, not from a converged model
2. The claimed superiority of FedAgent-Chain (F1=0.6374 < 0.6762) is comparing a fairly-trained model against the BEST ROUND of a diverging model — this is not a fair comparison
3. The catastrophic cross-seed variance (F1 range: 0.09–0.68 for Standard FedAvg) is directly caused by the divergence being seed-sensitive

### Root Cause Explanation
In `src/federated/client.py`, `train_round()` currently returns the model DELTA:
```python
delta = {
    name: final_state[name] - initial_state[name]
    for name in final_state
}
```

And in `src/federated/server.py`, `run_round()` applies this delta:
```python
self.global_state[name] = self.global_state[name] + aggregated_delta[name]
```

The problem is: `protect_state_dict()` adds Gaussian DP noise `N(0, (σ·C)²I)` to each parameter's delta. With `σ=0.1, C=1.0`, the noise std = 0.1 per parameter. With 4 nodes and 20 rounds, each parameter accumulates 80 noisy delta contributions. Since the DP noise has zero mean but is not zero-variance, the expected squared L2 norm of the accumulated noise grows as `O(T · K · σ² · C² · d)` where T=rounds, K=nodes, d=parameter dimension. For a model with ~50,000 parameters, this is an enormous unbounded term.

Standard FedAvg is defined as averaging ABSOLUTE model weights, not accumulating deltas. The delta formulation only makes sense in protocols like gradient-based FL (FedSGD) where deltas are true gradient updates, not in parameter-space averaging.

### Files Involved
- `src/federated/client.py` — `train_round()` must return absolute weights, not deltas
- `src/federated/server.py` — `run_round()` must apply weighted average of absolute weights
- `src/federated/aggregator.py` — `FedAvgAggregator.aggregate()` signature must reflect absolute weights
- `src/federated/privacy.py` — DP must be applied at the correct point (to the delta for communication privacy, not to the accumulated global state)
- `tests/unit/test_aggregator.py` — update tests to verify absolute weight aggregation
- `tests/integration/test_federated_pipeline.py` — update to verify convergence
- `tests/unit/test_privacy.py` — add delta-then-aggregate test

### Architectural Considerations

The canonical FedAvg algorithm (McMahan et al., 2017) works as follows:
1. Server broadcasts global weights `w_t` to all clients
2. Each client k trains locally for E epochs, producing local weights `w_k^{t+1}`
3. Server computes: `w^{t+1} = Σ_k (n_k / N) · w_k^{t+1}`

There is no delta in step 3. The aggregation is a weighted mean of absolute weights.

When DP protection is applied (as in DP-FedAvg), the protection is applied to the **communication artifact** (the delta `Δw_k = w_k^{t+1} - w_t`), and then the server reconstructs approximate absolute weights as `w_k^{perturbed} = w_t + Δw̃_k`. The aggregation is then:
`w^{t+1} = Σ_k (n_k / N) · w_k^{perturbed} = w_t + Σ_k (n_k / N) · Δw̃_k`

This formulation is EQUIVALENT to delta aggregation BUT ONLY IF `w_t` is subtracted before aggregation and added back after. The current code skips the subtraction — it aggregates the raw perturbed delta and adds it directly, which over multiple rounds becomes cumulative unbounded drift.

The cleanest fix: clients return the DP-protected ABSOLUTE weight vector (after adding noise to the delta, reconstruct the absolute weights). The server then aggregates absolute weights. This is mathematically identical to DP-FedAvg but eliminates the accumulation bug.

### Required Code Modifications

**Modification 1.1 — `src/federated/client.py`, `train_round()` return value**

```python
# BEFORE — returns raw delta (causes accumulation divergence)
delta: Dict[str, np.ndarray] = {
    name: final_state[name] - initial_state[name]
    for name in final_state
}
flat_delta = np.concatenate([v.flatten() for v in delta.values()])
protected_delta = protect_state_dict(delta, C=self.C, sigma=self.sigma, seed=seed)

# ... submit hash ...
return protected_delta, len(self.train_dataset), metrics

# AFTER — returns DP-protected ABSOLUTE weights (eliminates accumulation divergence)
# Step 1: Compute per-parameter delta
delta: Dict[str, np.ndarray] = {
    name: final_state[name] - initial_state[name]
    for name in final_state
}

# Step 2: Apply DP protection to delta (clipping + noise applied to delta, not absolute weights)
protected_delta = protect_state_dict(delta, C=self.C, sigma=self.sigma, seed=seed)

# Step 3: Reconstruct DP-protected absolute weights for communication to server
# w_k_protected = w_global + Δw̃_k
protected_absolute_weights: Dict[str, np.ndarray] = {
    name: initial_state[name] + protected_delta[name]
    for name in initial_state
}

# Step 4: Hash the delta (not absolute weights) for blockchain privacy audit
# The blockchain receives only the hash of the delta, protecting weight values
flat_protected_delta = np.concatenate([v.flatten() for v in protected_delta.values()])
self.blockchain.submit_model_update_hash(
    protected_update=flat_protected_delta,
    node_id=self.node_id,
    round_number=round_number,
    consent_ref=self.consent_ref,
    policy_ref=self.policy_ref,
)

metrics = self._evaluate_local(model)
metrics["train_loss"] = float(np.mean(epoch_losses))

# Return absolute weights (not delta) so server can perform true FedAvg mean
return protected_absolute_weights, len(self.train_dataset), metrics
```

**Modification 1.2 — `src/federated/server.py`, `run_round()` aggregation**

```python
# BEFORE — applies delta to global state (accumulation divergence)
aggregated_delta = self.aggregator.aggregate(updates)
for name in self.global_state:
    if name in aggregated_delta:
        self.global_state[name] = self.global_state[name] + aggregated_delta[name]

# AFTER — replaces global state with weighted mean of absolute weights (true FedAvg)
# updates now contains (node_id, absolute_weights, n_samples)
aggregated_weights = self.aggregator.aggregate(updates)
# Directly replace global state — no delta addition
for name in self.global_state:
    if name in aggregated_weights:
        self.global_state[name] = aggregated_weights[name].copy()
    else:
        self.logger.warning(
            "Parameter missing from aggregation result",
            param_name=name,
            available_keys=list(aggregated_weights.keys()),
        )
```

**Modification 1.3 — `src/federated/server.py`, add convergence diagnostics**

After `run_round()` in the round loop, add gradient norm and weight norm logging:
```python
# Convergence diagnostics — add after aggregation
total_weight_norm = float(np.sqrt(sum(
    np.sum(v ** 2) for v in self.global_state.values()
)))
self.logger.info(
    "Global model weight norm after aggregation",
    round=round_num,
    weight_norm=round(total_weight_norm, 4),
)
# Loss explosion detection — halt training if loss exceeds threshold
if round_metrics.get("mean_train_loss", 0.0) > 50.0:
    self.logger.error(
        "LOSS EXPLOSION DETECTED — halting training",
        round=round_num,
        mean_train_loss=round_metrics.get("mean_train_loss"),
    )
    raise RuntimeError(
        f"Training halted: mean_train_loss={round_metrics['mean_train_loss']:.2f} "
        f"exceeds explosion threshold of 50.0 at round {round_num}. "
        "Check FedAvg aggregation logic and DP noise parameters."
    )
```

**Modification 1.4 — `src/federated/aggregator.py`, update docstrings and type hints**

Update all docstrings in `FedAvgAggregator.aggregate()` and `FairnessAwareFedAvgAggregator.aggregate()` to explicitly state that the input `state_dict` entries represent **absolute model weights**, not deltas. Add an assertion:
```python
# At the top of aggregate():
# Defensive assertion: weights should not contain NaN or Inf (which would indicate DP noise corruption)
for node_id, state_dict, _ in updates:
    for param_name, param_val in state_dict.items():
        if np.any(np.isnan(param_val)) or np.any(np.isinf(param_val)):
            raise ValueError(
                f"NaN or Inf detected in state_dict from node '{node_id}', "
                f"parameter '{param_name}'. This indicates a DP noise or "
                f"weight accumulation bug. Halting aggregation."
            )
```

**Modification 1.5 — `src/federated/privacy.py`, add DP diagnostics**

Add per-call logging to `protect_state_dict()`:
```python
# After computing protected, add:
original_norms = {name: float(np.linalg.norm(state_dict[name])) for name in state_dict}
protected_norms = {name: float(np.linalg.norm(protected[name])) for name in protected}
logger.debug(
    "DP protection applied to state dict",
    n_params=len(state_dict),
    max_original_norm=max(original_norms.values()),
    max_protected_norm=max(protected_norms.values()),
    C=C,
    sigma=sigma,
)
```

### Required Tests

**Test 1.A — Unit test: aggregation of absolute weights**
Add to `tests/unit/test_aggregator.py`:
```python
def test_aggregate_converges_not_diverges():
    """Verify that applying FedAvg aggregation 20 times does not cause weight norm explosion."""
    agg = FedAvgAggregator()
    rng = np.random.default_rng(42)
    
    # Initialize global state
    global_state = {
        "layer.weight": rng.standard_normal((64, 91)).astype(np.float32),
        "layer.bias": rng.standard_normal(64).astype(np.float32),
    }
    
    initial_norm = np.sqrt(sum(np.sum(v**2) for v in global_state.values()))
    
    # Simulate 20 rounds of FedAvg aggregation
    for round_num in range(20):
        updates = []
        for i in range(4):
            # Simulate local training: add small perturbation to global state
            local_state = {
                name: val + rng.standard_normal(val.shape).astype(np.float32) * 0.01
                for name, val in global_state.items()
            }
            updates.append((f"node_{i}", local_state, 1000))
        
        global_state = agg.aggregate(updates)
    
    final_norm = np.sqrt(sum(np.sum(v**2) for v in global_state.values()))
    
    # Weight norm should not explode (allow 5x growth as generous bound)
    assert final_norm < initial_norm * 5.0, (
        f"Weight norm exploded from {initial_norm:.4f} to {final_norm:.4f} "
        f"over 20 FedAvg rounds — indicates delta accumulation bug"
    )
```

**Test 1.B — Integration test: loss does not diverge over 20 rounds**
Add to `tests/integration/test_federated_pipeline.py`:
```python
@pytest.mark.integration
def test_fedavg_loss_does_not_diverge(small_cfg, two_node_clients):
    """Critical: FedAvg loss must not increase monotonically (indicates delta accumulation bug)."""
    small_cfg.federated.n_rounds = 10
    small_cfg.federated.local_epochs = 2
    small_cfg.federated.min_clients = 2
    server = FederatedServer(small_cfg, use_fairness_aggregation=False, output_dir="/tmp/test_convergence")
    
    round_losses = []
    for rnd in range(1, 11):
        metrics = server.run_round(two_node_clients[0], round_num=rnd, seed=42)
        round_losses.append(metrics.get("mean_train_loss", float("inf")))
    
    # Loss at round 10 must not be more than 5x the loss at round 1
    loss_ratio = round_losses[-1] / (round_losses[0] + 1e-8)
    assert loss_ratio < 5.0, (
        f"FedAvg loss diverged: round 1={round_losses[0]:.4f}, "
        f"round 10={round_losses[-1]:.4f}, ratio={loss_ratio:.2f}. "
        f"Full history: {[round(x, 4) for x in round_losses]}"
    )
    
    # Additionally, final loss should not exceed 5.0 for binary classification
    assert round_losses[-1] < 5.0, (
        f"FedAvg BCE loss={round_losses[-1]:.4f} at round 10 is unreasonably high "
        f"for binary classification. Expected < 5.0."
    )
```

**Test 1.C — Regression test: weight norm stays bounded**
```python
@pytest.mark.regression
def test_global_model_weight_norm_bounded():
    """Verify that the global model weight norm remains within reasonable bounds after 20 rounds."""
    # After running the full simulation (seed=42, 20 rounds), load the final checkpoint
    # and verify weight norm is reasonable for a model trained on binary classification
    ckpt_path = Path("experiments/runs/") / ... # find most recent fedagent_chain_full run
    # ...
    total_norm = ... # compute weight norm
    assert total_norm < 1000.0, f"Weight norm {total_norm} is unreasonably large"
    assert total_norm > 0.01, f"Weight norm {total_norm} is near zero (collapsed model)"
```

### Validation Methodology

After implementing the fix, run the no_fairness ablation for 20 rounds and verify the per_round.json:
```bash
# Generate fresh data
python scripts/generate_synthetic_data.py --config configs/experiment/fedagent_chain_full.yaml --seed 42

# Run no_fairness ablation (Standard FedAvg)
python scripts/run_federated_simulation.py \
    --config configs/experiment/ablation/no_fairness.yaml \
    --seed 42 --no-mlflow

# Check the per_round.json for the new run
python -c "
import json; import sys
from pathlib import Path
runs = sorted(Path('experiments/runs').glob('ablation_no_fairness_seed42_*'))
latest = runs[-1] / 'metrics' / 'per_round.json'
with open(latest) as f:
    history = json.load(f)
losses = [r['mean_train_loss'] for r in history]
print('Per-round BCE losses:', [round(x, 4) for x in losses])
final_loss = losses[-1]
assert final_loss < 2.0, f'FAIL: Final loss {final_loss:.4f} > 2.0, FedAvg still diverging'
assert losses[-1] < losses[0] * 3, f'FAIL: Loss at round 20 is {losses[-1]/losses[0]:.1f}x round 1 loss'
print('PASS: Loss convergence verified')
"
```

### Logging Requirements
- Every `train_round()` call must log: `train_loss`, `f1`, `weight_norm_delta`, `dp_noise_applied`
- Every `run_round()` call must log: `global_weight_norm`, `aggregated_weight_norm`, loss explosion check result
- If weight norm exceeds 500.0 at any round, emit a WARNING level log

### Expected Outputs
- Standard FedAvg 20-round per_round.json showing stable or decreasing BCE loss
- BCE loss at round 20 < 2.0 (reasonable for binary classification with sigmoid)
- No `RuntimeError` from loss explosion detector
- All unit tests in `test_aggregator.py` passing
- Integration convergence test passing

### Success Criteria
- [ ] `tests/unit/test_aggregator.py::test_aggregate_converges_not_diverges` PASSES
- [ ] `tests/integration/test_federated_pipeline.py::test_fedavg_loss_does_not_diverge` PASSES
- [ ] `tests/integration/test_federated_pipeline.py::test_global_model_state_updates_after_round` PASSES
- [ ] Full 20-round ablation no_fairness run: BCE loss at round 20 < 2.0
- [ ] Full 20-round ablation no_fairness run: BCE loss ratio (round20/round1) < 3.0
- [ ] Global weight norm at round 20 < 500.0
- [ ] No NaN or Inf in any checkpoint parameter

### Failure Criteria
- Loss at round 20 > 5.0 (still diverging)
- Any NaN in checkpoint state dict
- Weight norm ratio > 10.0 (still exploding)
- Any new test failures introduced that did not exist in Phase 0 baseline

### Stop Conditions
If loss is still monotonically increasing after the fix, halt. The aggregation logic must be inspected again. Print the global weight norm at every round and the aggregated weight norm before and after application. The issue may be in `protect_state_dict` applying excessively large DP noise to the absolute weights rather than to deltas only.

### Git Commit Recommendation
```
git add src/federated/client.py src/federated/server.py src/federated/aggregator.py \
        src/federated/privacy.py tests/unit/test_aggregator.py \
        tests/integration/test_federated_pipeline.py
git commit -m "phase1: fix FedAvg delta-accumulation divergence, add convergence diagnostics

- client returns DP-protected absolute weights (not raw delta)
- server applies weighted mean of absolute weights (true FedAvg)
- aggregator validates no NaN/Inf in input state dicts
- add loss explosion detector (RuntimeError if loss > 50)
- add per-round weight norm logging
- add convergence unit test (20 rounds, norm stays bounded)
- add integration test (loss at round 10 < 5x round 1 loss)

Fixes: CI-2 (FedAvg divergence), prerequisite for CI-3 (cross-seed variance)"
```

---

**PHASE COMPLETION CHECKLIST — PHASE 1**

**SUCCESS CONDITIONS:**
- All new tests pass ✓
- No regressions from Phase 0 baseline ✓
- No_fairness 20-round loss < 2.0 at round 20 ✓
- Weight norm bounded ✓

**FAIL CONDITIONS:**
- Loss still monotonically increasing after fix
- New NaN in checkpoints
- Any test_aggregator.py test newly failing

**NEXT PHASE ENTRY REQUIREMENTS:**
Phase 1 success criteria verified. `per_round.json` for no_fairness seed=42 shows non-diverging loss. Unit and integration tests pass.

**DO NOT CONTINUE TO PHASE 2 UNTIL ALL SUCCESS CRITERIA PASS.**

---

## Phase 2 — Fairness Pipeline Corrections

### Objective
Fix the `_get_batch_group_labels` index mismatch bug that causes the fairness penalty Ω_fair to be computed on incorrect group-sample pairings. This is the likely root cause of FedAgent-Chain failing to improve (and in some attributes worsening) fairness disparity. After this fix, re-run the fairness training and verify that D_fair for disability actually decreases with fairness-aware FedAvg compared to standard FedAvg.

### Why This Phase Matters
The entire scientific contribution of FedAgent-Chain is built on the claim that fairness-aware federated training reduces disparity across protected groups. The audit found that disability D_fair *increases* by 106% with FedAgent-Chain vs Standard FedAvg. If this is caused by the group label mismatch (fairness penalty being applied to wrong groups, producing adversarial gradients that harm minority groups), then fixing this bug should restore the fairness improvement.

If the fairness disparity still increases AFTER this fix, then the theoretical mechanism of the fairness penalty needs fundamental redesign — which would require a major revision of Section 4.6 of the paper. This phase determines which scenario is true.

### Root Cause Explanation
In `src/federated/client.py`, `_get_batch_group_labels()`:
```python
def _get_batch_group_labels(self, batch: dict) -> np.ndarray | None:
    if not hasattr(self, "_group_label_cache"):
        try:
            self._group_label_cache = self.train_dataset.get_group_labels("disability_category")
        except Exception:
            self._group_label_cache = None

    if self._group_label_cache is None:
        return None

    # BUG: Returns first N items of full-dataset cache, not the actual batch items
    return self._group_label_cache[: len(batch["label"])]
```

The `DataLoader` with `shuffle=True` reorders samples. `batch["label"]` contains labels for items at positions `[idx_0, idx_1, ..., idx_N-1]` where these indices are shuffled. But `self._group_label_cache[:N]` returns group labels for dataset positions `[0, 1, ..., N-1]` — a completely different set of items. The fairness penalty is thus computed between predictions for items from one position and group memberships from entirely different items.

This is especially harmful because the fairness penalty tries to equalize predictions across groups, but with the wrong group assignments, it may actually WIDEN disparities.

### Files Involved
- `src/data/dataset.py` — `EmploymentDataset.__getitem__` must return sample index
- `src/federated/client.py` — `_get_batch_group_labels` must use batch indices
- `tests/unit/test_fairness.py` — add batch alignment test
- `tests/integration/test_federated_pipeline.py` — add fairness penalty correctness test

### Required Code Modifications

**Modification 2.1 — `src/data/dataset.py`, `__getitem__` returns index**

```python
# BEFORE
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    row = self.outcomes.iloc[idx]
    # ... encoding logic ...
    item: Dict[str, torch.Tensor] = {
        "features": torch.from_numpy(features),
        "label": torch.tensor(label, dtype=torch.float32),
    }
    if self.sample_weights is not None:
        item["weight"] = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32)
    return item

# AFTER — include dataset index for correct group label lookup
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    row = self.outcomes.iloc[idx]
    # ... encoding logic (unchanged) ...
    item: Dict[str, torch.Tensor] = {
        "features": torch.from_numpy(features),
        "label": torch.tensor(label, dtype=torch.float32),
        "idx": torch.tensor(idx, dtype=torch.long),  # ADD THIS LINE
    }
    if self.sample_weights is not None:
        item["weight"] = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32)
    return item
```

**Modification 2.2 — `src/federated/client.py`, `_get_batch_group_labels` corrected**

```python
# BEFORE — returns wrong indices
def _get_batch_group_labels(self, batch: dict) -> np.ndarray | None:
    if not hasattr(self, "_group_label_cache"):
        try:
            self._group_label_cache = self.train_dataset.get_group_labels("disability_category")
        except Exception:
            self._group_label_cache = None
    if self._group_label_cache is None:
        return None
    return self._group_label_cache[: len(batch["label"])]  # BUG

# AFTER — uses actual batch indices for correct group lookup
def _get_batch_group_labels(self, batch: dict) -> np.ndarray | None:
    """Return disability_category group labels for the exact samples in this batch.
    
    Uses the dataset index stored in batch['idx'] (added in EmploymentDataset.__getitem__)
    to look up the correct group labels for each sample in the shuffled batch.
    
    Returns None gracefully if group labels are unavailable.
    """
    if not hasattr(self, "_group_label_cache"):
        try:
            self._group_label_cache = self.train_dataset.get_group_labels("disability_category")
        except Exception:
            self.logger.warning(
                "Failed to build group label cache for fairness penalty",
                node_id=self.node_id,
            )
            self._group_label_cache = None
    
    if self._group_label_cache is None:
        return None
    
    batch_indices = batch.get("idx")
    if batch_indices is None:
        self.logger.warning(
            "Batch does not contain 'idx' key — fairness penalty cannot be applied. "
            "Ensure EmploymentDataset.__getitem__ returns 'idx'.",
            node_id=self.node_id,
        )
        return None
    
    indices = batch_indices.numpy().astype(int)
    
    # Bounds check
    max_idx = len(self._group_label_cache) - 1
    if np.any(indices > max_idx) or np.any(indices < 0):
        self.logger.error(
            "Batch indices out of bounds for group label cache",
            max_cache_idx=max_idx,
            batch_idx_range=(int(indices.min()), int(indices.max())),
            node_id=self.node_id,
        )
        return None
    
    return self._group_label_cache[indices]
```

**Modification 2.3 — `src/federated/client.py`, improve fairness penalty logging**

In the `train_round()` training loop, when the fairness penalty fires, log:
```python
if fairness_enabled and batch_counter % 5 == 0:
    try:
        batch_group_labels = self._get_batch_group_labels(batch)
        if batch_group_labels is not None:
            # Verify alignment: same number of labels as batch samples
            assert len(batch_group_labels) == len(batch["label"]), (
                f"Group label count ({len(batch_group_labels)}) != "
                f"batch size ({len(batch['label'])})"
            )
            # ... existing penalty computation ...
            if len(group_means) >= 2:
                vals = torch.stack(list(group_means.values()))
                diff = vals.max() - vals.min()
                fairness_penalty_value = float(diff.item())
                loss = loss + lambda_fair * diff
                
                # Log fairness penalty magnitude every 50 batches
                if batch_counter % 50 == 0:
                    self.logger.debug(
                        "Fairness penalty applied",
                        node_id=self.node_id,
                        batch=batch_counter,
                        n_groups_in_batch=len(group_means),
                        fairness_penalty=round(fairness_penalty_value, 6),
                        lambda_fair=lambda_fair,
                    )
    except AssertionError as e:
        self.logger.error("Group label alignment failed", error=str(e))
        # Do not apply fairness penalty if alignment fails
```

**Modification 2.4 — Add fairness penalty unit test for alignment correctness**

Add to `tests/unit/test_fairness.py`:
```python
def test_batch_group_labels_alignment():
    """Verify that group labels returned by _get_batch_group_labels 
    correspond to the correct samples via the idx field."""
    from src.data.dataset import EmploymentDataset
    from src.data.synthetic_generator import generate_synthetic_node_data
    from torch.utils.data import DataLoader
    
    data = generate_synthetic_node_data("saudi_arabia", 100, 50, 200, seed=42)
    ds = EmploymentDataset(data["outcomes"], data["users"], data["jobs"])
    
    # Verify idx is returned
    item = ds[0]
    assert "idx" in item, "EmploymentDataset.__getitem__ must return 'idx'"
    assert item["idx"].item() == 0
    
    item = ds[42]
    assert item["idx"].item() == 42
    
    # Verify idx survives DataLoader batching
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    assert "idx" in batch, "DataLoader must propagate 'idx' field"
    assert len(batch["idx"]) == 16
    
    # Verify group labels correspond to correct samples
    group_cache = ds.get_group_labels("disability_category")
    batch_indices = batch["idx"].numpy()
    expected_groups = group_cache[batch_indices]
    
    # Manually simulate what _get_batch_group_labels now does
    actual_groups = group_cache[batch_indices]
    
    assert list(expected_groups) == list(actual_groups), (
        "Group labels do not match expected values for batch indices"
    )
    
    # VERIFY THE OLD BUG: check that first-N-items approach gives DIFFERENT results
    old_approach = group_cache[:16]
    # With shuffling, these should differ from correct alignment (with high probability)
    # This is not guaranteed but very likely with shuffled data
    match_count = sum(a == b for a, b in zip(old_approach, actual_groups))
    print(f"Old approach matches correct labels for {match_count}/16 samples (expected < 16 with shuffling)")
```

### Fairness Validation Protocol

After the fix, run the fairness-aware simulation and measure actual D_fair before and after:
```bash
# Step 1: Run Standard FedAvg (no fairness) with fixed FedAvg
python scripts/run_federated_simulation.py \
    --config configs/experiment/ablation/no_fairness.yaml \
    --seed 42 --no-mlflow

# Step 2: Run FedAgent-Chain (with fairness) with fixed group labels
python scripts/run_federated_simulation.py \
    --config configs/experiment/fedagent_chain_full.yaml \
    --seed 42 --no-mlflow

# Step 3: Evaluate both and compare D_fair
python scripts/run_evaluation.py \
    --runs-dir experiments/runs/ \
    --results-dir experiments/results/phase2_fairness_check/ \
    --seed 42

# Step 4: Verify D_fair(disability) is LOWER for FedAgent-Chain than Standard FedAvg
python -c "
import pandas as pd
df = pd.read_csv('experiments/results/phase2_fairness_check/table_3_fairness_results.csv')
fa_disability = df.loc[df['Attribute'].str.contains('Disab'), 'FedAgent-Chain'].values[0]
std_fl_disability = df.loc[df['Attribute'].str.contains('Disab'), 'Standard FedAvg'].values[0]
print(f'FedAgent-Chain disability D_fair: {fa_disability:.4f}')
print(f'Standard FedAvg disability D_fair: {std_fl_disability:.4f}')
if fa_disability < std_fl_disability:
    print('PASS: FedAgent-Chain reduces disability disparity')
else:
    print(f'FAIL: FedAgent-Chain still does NOT reduce disability disparity')
    print('This requires redesign of the fairness regularization objective')
    print('See architectural note in REPAIR_LOG.md for alternative approaches')
"
```

**CRITICAL DECISION POINT**: If D_fair(disability) for FedAgent-Chain is still ≥ Standard FedAvg after this fix, halt and evaluate whether:
1. The λ_fairness value (0.5) is too large (causing overcorrection) — try λ ∈ {0.05, 0.1, 0.2}
2. The surrogate penalty (mean predicted probability difference) is insufficient — consider F1 disparity surrogate
3. The Non-IID data distribution makes group-level fairness improvement fundamentally difficult at this data scale

Document findings in `REPAIR_LOG.md` before continuing.

### Logging Requirements
- Log group distribution in each batch (number of samples per group) every 100 batches
- Log fairness penalty magnitude every 50 batches
- Log D_fair at every round for all 4 protected attributes

### Expected Outputs
- `experiments/results/phase2_fairness_check/table_3_fairness_results.csv` with corrected D_fair values
- D_fair(disability) for FedAgent-Chain ≤ D_fair for Standard FedAvg (OR documented explanation for why it does not)

### Success Criteria
- [ ] `tests/unit/test_fairness.py::test_batch_group_labels_alignment` PASSES
- [ ] `EmploymentDataset.__getitem__` returns `idx` key
- [ ] `_get_batch_group_labels` uses `batch["idx"]` indices
- [ ] D_fair(disability): FedAgent-Chain < Standard FedAvg, OR documented redesign proposal in REPAIR_LOG.md
- [ ] No new test failures

### Failure Criteria
- `batch["idx"]` not returned by DataLoader (check custom collate_fn interference)
- D_fair(disability) still higher for FedAgent-Chain AND no redesign plan documented

### Stop Conditions
If `batch["idx"]` is not propagated by PyTorch DataLoader (which would indicate a custom collate_fn issue), stop and investigate the DataLoader configuration. Default DataLoader collates tensor fields automatically, so this should work without changes.

### Git Commit Recommendation
```
git add src/data/dataset.py src/federated/client.py \
        tests/unit/test_fairness.py
git commit -m "phase2: fix group label alignment in fairness penalty

- EmploymentDataset.__getitem__ now returns 'idx' (dataset position)
- _get_batch_group_labels uses batch['idx'] for correct sample-to-group mapping
- adds alignment assertion to catch future regressions
- adds fairness penalty magnitude logging
- fixes root cause of D_fair(disability) increasing with fairness-aware training

Fixes: CI-5 (group label mismatch)"
```

---

**PHASE COMPLETION CHECKLIST — PHASE 2**

**SUCCESS CONDITIONS:**
- Alignment unit test passes ✓
- idx propagated through DataLoader ✓
- Fairness results show directional improvement OR redesign documented ✓

**FAIL CONDITIONS:**
- idx not in batch dict
- D_fair still worsens without documented redesign plan

**NEXT PHASE ENTRY REQUIREMENTS:**
Phase 2 success criteria met. Table 3 generated with corrected alignment.

**DO NOT CONTINUE TO PHASE 3 UNTIL ALL SUCCESS CRITERIA PASS.**

---

## Phase 3 — Baseline Integrity Repairs

### Objective
Fix two baseline integrity issues: (1) the Centralized baseline flag is never handled in the data loading pipeline, causing Local Baseline == Centralized outputs, and (2) fix the dead code in `_find_best_checkpoint` that makes the fallback logic unreachable. Additionally, fix `run_baselines.py` to not manipulate `sys.argv`. Verify that Local Baseline and Centralized produce genuinely different outputs.

### Why This Phase Matters
Table 2 in the paper contains four methods: FedAgent-Chain, Standard FedAvg, Local Baseline, and Centralized. Two of these four (Local Baseline and Centralized) are currently producing identical outputs, meaning 50% of the comparison table is meaningless. The Centralized baseline is the theoretical upper bound on FL performance — it represents "if we could pool all data at one location, how well would we do?" This is a critical anchor for interpreting FL results. Without it, we cannot assess whether federation recovers most of the utility lost by data locality.

### Root Cause Explanation

**Root Cause 3A: Centralized flag not handled**
In `scripts/run_federated_simulation.py`, `load_node_dataset()`:
```python
def load_node_dataset(node_id, data_dir, cfg, seed):
    # Only loads per-node data regardless of cfg
    users_csv = node_dir / "users.csv"
    # ...
    full_dataset = EmploymentDataset(outcomes_df, users_df, jobs_df, consent_filter=True)
    train_ds, test_ds = full_dataset.split(test_size=0.20, seed=seed)
    return train_ds, test_ds
```
The `data.centralized: true` flag in `configs/experiment/baseline_centralized.yaml` is never read. Centralized training should pool all four nodes' data into a single training set and train a single model with no federation.

**Root Cause 3B: Dead code after return in `_find_best_checkpoint`**
In `scripts/run_evaluation.py`:
```python
def _find_best_checkpoint(run_dir: Path) -> Path:
    # ... logic ...
    candidate = ckpt_dir / f"global_model_round_{best_round:03d}.pt"
    logger.info("Selected best checkpoint", ...)
    return candidate   # <-- EARLY RETURN

    # DEAD CODE: Never executed
    saved = sorted(ckpt_dir.glob("global_model_round_*.pt"))
    # ... fallback logic ...
```

**Root Cause 3C: `run_baselines.py` sys.argv manipulation**
`run_baselines.py` overwrites `sys.argv` to call `run_main()`, which is fragile and will fail in test environments.

### Files Involved
- `scripts/run_federated_simulation.py` — `load_node_dataset()` must handle centralized flag
- `scripts/run_evaluation.py` — `_find_best_checkpoint()` dead code removal + fallback fix
- `scripts/run_baselines.py` — sys.argv manipulation removal
- `configs/experiment/baseline_centralized.yaml` — verify configuration
- `tests/unit/test_config_loading.py` — add centralized mode test

### Required Code Modifications

**Modification 3.1 — `scripts/run_federated_simulation.py`, implement centralized mode**

```python
def load_node_dataset(node_id: str, data_dir: Path, cfg: object, seed: int) -> tuple:
    """Load or generate dataset for a single node and return train/test split.
    
    When cfg.data.centralized is True, this function is called with node_id='centralized'
    and returns data pooled from all four regional nodes. This implements the 
    centralized baseline where all data is available to a single model.
    """
    data_cfg = cfg.get("data", {})
    is_centralized = bool(data_cfg.get("centralized", False))
    
    if is_centralized:
        # Load ALL nodes' data and concatenate
        # This represents the theoretical upper bound: if data sovereignty could be waived
        all_nodes = data_cfg.get("nodes", ["saudi_arabia", "united_states", "china", "europe"])
        all_users = []
        all_jobs = []
        all_outcomes = []
        
        for nid in all_nodes:
            nid_dir = data_dir / nid
            if not (nid_dir / "users.csv").exists():
                raise FileNotFoundError(
                    f"Centralized baseline: data for node '{nid}' not found at {nid_dir}. "
                    "Run generate_synthetic_data.py first."
                )
            all_users.append(pd.read_csv(nid_dir / "users.csv"))
            all_jobs.append(pd.read_csv(nid_dir / "jobs.csv"))
            all_outcomes.append(pd.read_csv(nid_dir / "outcomes.csv"))
        
        users_df = pd.concat(all_users, ignore_index=True)
        jobs_df = pd.concat(all_jobs, ignore_index=True)
        outcomes_df = pd.concat(all_outcomes, ignore_index=True)
        
        logger.info(
            "Centralized baseline: pooled data from all nodes",
            n_nodes=len(all_nodes),
            total_users=len(users_df),
            total_jobs=len(jobs_df),
            total_outcomes=len(outcomes_df),
        )
    else:
        # Standard per-node loading (existing logic)
        node_dir = data_dir / node_id
        # ... existing per-node loading code ...
    
    full_dataset = EmploymentDataset(
        outcomes_df=outcomes_df,
        users_df=users_df,
        jobs_df=jobs_df,
        consent_filter=True,
    )
    train_ds, test_ds = full_dataset.split(test_size=0.20, seed=seed)
    return train_ds, test_ds
```

Also update `main()` in `run_federated_simulation.py` to handle centralized mode:
```python
is_centralized = bool(cfg.get("data", {}).get("centralized", False))

if is_centralized:
    # For centralized baseline: single client, single node, all data pooled
    train_ds, test_ds = load_node_dataset("centralized", data_dir, cfg, seed=args.seed)
    clients = [FederatedClient(
        node_id="centralized",
        train_dataset=train_ds,
        test_dataset=test_ds,
        cfg=cfg,
        blockchain=blockchain,
        device="cpu",
    )]
    # Override min_clients for centralized mode
    cfg.federated.min_clients = 1
    logger.info("Centralized mode: single client with pooled data", total_samples=len(train_ds))
else:
    # Standard multi-node loading
    clients = []
    for i, node_id in enumerate(nodes):
        # ... existing logic ...
```

**Modification 3.2 — `scripts/run_evaluation.py`, fix dead code in `_find_best_checkpoint`**

```python
def _find_best_checkpoint(run_dir: Path) -> Path:
    """Return path to the best-performing checkpoint by F1, or final_model.pt.
    
    Selects the checkpoint at the round with highest mean_f1.
    If no per-round checkpoints exist, falls back to final_model.pt.
    If neither exists, raises FileNotFoundError.
    """
    per_round_path = run_dir / "metrics" / "per_round.json"
    final_path = run_dir / "checkpoints" / "final_model.pt"
    ckpt_dir = run_dir / "checkpoints"

    if not per_round_path.exists():
        logger.warning("per_round.json not found, using final_model.pt", run_dir=str(run_dir))
        return final_path

    with open(per_round_path, encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        logger.warning("per_round.json is empty, using final_model.pt", run_dir=str(run_dir))
        return final_path

    # Find rounds that have checkpoint files
    valid_history = []
    for r in history:
        r_num = r.get("round")
        if r_num is not None and (ckpt_dir / f"global_model_round_{r_num:03d}.pt").exists():
            valid_history.append(r)

    if not valid_history:
        logger.warning(
            "No per-round checkpoints found, using final_model.pt",
            run_dir=str(run_dir),
            history_rounds=[r.get("round") for r in history],
        )
        return final_path

    # Select best round by mean_f1 (ties broken by later round for more stable weights)
    best_round_data = max(valid_history, key=lambda r: (r.get("mean_f1", 0.0), r.get("round", 0)))
    best_round = best_round_data.get("round")
    candidate = ckpt_dir / f"global_model_round_{best_round:03d}.pt"
    
    # Verify candidate exists (it should since we checked above)
    if not candidate.exists():
        # THIS IS THE FALLBACK — now reachable because we removed the premature return
        logger.warning(
            "Best round checkpoint disappeared, falling back to nearest",
            best_round=best_round,
            run_dir=str(run_dir),
        )
        saved_rounds = []
        for p in sorted(ckpt_dir.glob("global_model_round_*.pt")):
            try:
                saved_rounds.append((int(p.stem.split("_")[-1]), p))
            except ValueError:
                pass
        
        if not saved_rounds:
            return final_path
        
        # Find nearest saved checkpoint at or before best_round
        below = [(r, p) for r, p in saved_rounds if r <= best_round]
        if below:
            rnd, path = max(below, key=lambda x: x[0])
            logger.info("Using nearest saved checkpoint", round=rnd, requested_best=best_round)
            return path
        
        # Use earliest available if nothing below
        rnd, path = min(saved_rounds, key=lambda x: x[0])
        logger.info("Using earliest available checkpoint", round=rnd)
        return path
    
    logger.info(
        "Selected best checkpoint",
        round=best_round,
        f1=round(best_round_data.get("mean_f1", 0.0), 4),
        run_dir=run_dir.name,
    )
    return candidate
```

**Modification 3.3 — `scripts/run_baselines.py`, remove sys.argv manipulation**

```python
# BEFORE — fragile sys.argv manipulation
from scripts.run_federated_simulation import main as run_main
import sys
sys.argv = ["run_federated_simulation.py", "--config", args.config, "--seed", str(args.seed), "--no-fairness", "--no-mlflow"]
if args.output_dir:
    sys.argv += ["--output-dir", args.output_dir]
run_main()

# AFTER — direct config loading and simulation call
from scripts.run_federated_simulation import run_simulation_from_config

cfg = load_config(args.config)
exp_name = cfg.get("experiment", {}).get("name", "baseline")
output_dir = Path(args.output_dir) if args.output_dir else Path(f"experiments/runs/{exp_name}_seed{args.seed}")

logger.info("Running baseline", experiment=exp_name, seed=args.seed)
run_simulation_from_config(
    cfg=cfg,
    seed=args.seed,
    output_dir=output_dir,
    use_fairness=False,
    use_mlflow=False,
)
```

Then refactor `scripts/run_federated_simulation.py` to expose:
```python
def run_simulation_from_config(cfg, seed, output_dir, use_fairness=True, use_mlflow=False):
    """Run the federated simulation given a config object (not sys.argv).
    
    This is the programmatic entry point used by run_baselines.py and run_ablation_study.py.
    It avoids sys.argv manipulation and is safe to call from test environments.
    """
    set_global_seed(seed)
    # ... rest of simulation logic extracted from main() ...
```

### Validation: Centralized ≠ Local Baseline

```bash
# Run local baseline
python scripts/run_baselines.py \
    --config configs/experiment/baseline_local.yaml --seed 42

# Run centralized baseline
python scripts/run_federated_simulation.py \
    --config configs/experiment/baseline_centralized.yaml --seed 42 --no-mlflow

# Run evaluation and verify they are NOT identical
python scripts/run_evaluation.py \
    --runs-dir experiments/runs/ \
    --results-dir experiments/results/phase3_baseline_check/ \
    --seed 42

python -c "
import pandas as pd
df = pd.read_csv('experiments/results/phase3_baseline_check/table_2_model_performance.csv')
local = df[df['Method'] == 'Local Baseline']['F1'].values[0]
central = df[df['Method'] == 'Centralized']['F1'].values[0]
print(f'Local Baseline F1:  {local:.4f}')
print(f'Centralized F1:     {central:.4f}')
print(f'Difference:         {abs(central - local):.4f}')
if abs(central - local) > 0.005:
    print('PASS: Centralized and Local Baseline are genuinely different')
else:
    print('FAIL: Centralized and Local Baseline still produce identical outputs')
    print('Check that centralized mode loads all 4 nodes data correctly')
assert central >= local - 0.05, (
    f'Centralized F1 ({central:.4f}) is substantially below Local Baseline ({local:.4f}). '
    f'This is unusual — centralized training should generally match or exceed local training '
    f'due to more available data. Investigate.'
)
"
```

### Success Criteria
- [ ] Local Baseline and Centralized produce metrics differing by > 0.005 on F1
- [ ] Centralized baseline trains on ~40,000 pairs (4 × 10,000 training) vs ~8,000 per local node
- [ ] Centralized F1 ≥ Local Baseline F1 - 0.05 (centralized should be at least as good as local)
- [ ] `_find_best_checkpoint` fallback code is reachable (no dead code)
- [ ] `run_baselines.py` does not manipulate `sys.argv`
- [ ] Unit test for centralized data loading passes

### Failure Criteria
- Local Baseline == Centralized outputs after fix → investigate data loading code
- Centralized F1 dramatically lower than local (by > 0.10) → indicates pooling creates train/test imbalance

### Git Commit Recommendation
```
git add scripts/run_federated_simulation.py scripts/run_evaluation.py \
        scripts/run_baselines.py tests/unit/
git commit -m "phase3: fix centralized baseline data pooling, fix dead code in _find_best_checkpoint

- load_node_dataset handles data.centralized=True by pooling all 4 nodes
- _find_best_checkpoint: remove dead code after premature return, fix fallback
- run_baselines.py: remove sys.argv manipulation, use run_simulation_from_config()
- expose run_simulation_from_config() as programmatic entry point

Fixes: CI-4 (Local == Centralized), M2 (dead code), M4 (sys.argv hack)"
```

---

**PHASE COMPLETION CHECKLIST — PHASE 3**

**SUCCESS CONDITIONS:**
- Centralized ≠ Local Baseline outputs ✓
- Dead code removed ✓
- sys.argv manipulation removed ✓

**FAIL CONDITIONS:**
- Centralized still produces identical output to Local

**NEXT PHASE ENTRY REQUIREMENTS:**
Phase 3 success criteria met. Centralized and Local produce genuinely different F1 values.

**DO NOT CONTINUE TO PHASE 4 UNTIL ALL SUCCESS CRITERIA PASS.**

---

## Phase 4 — Statistical and Evaluation Corrections

### Objective
Fix the numerically degenerate multi-seed statistical aggregation that produces t=-∞, p=0, Cohen's d=-38,800,000,000. Remove the invalid per-seed multi-seed summary files. Verify the main statistical tests produce valid numbers. Document the honest statistical interpretation: with 3 seeds and FedAgent-Chain F1 < Standard FedAvg F1 (before Phase 5 re-runs), the comparison may not be statistically significant.

### Why This Phase Matters
A paper claiming "our method is significantly better" when the statistical test shows p=0.2996 is scientifically dishonest. The numerically degenerate per-seed tests (t=-∞) are not just incorrect — they are statistically impossible values that any reviewer will immediately flag. This phase ensures all statistical claims are honest, correctly computed, and appropriately qualified.

### Root Cause Explanation
The `aggregate_multi_seed_results.py` script, when run from within a seed-specific subdirectory OR with only one seed's data available, loads a single per-seed CSV. It then creates a `table_2_multi_seed_summary.csv` that appears to have 3 seeds but actually has the same F1 value repeated 3 times (producing std=0). When `scipy.stats.ttest_rel` receives two arrays that are BOTH constant (all values identical), the test statistic becomes undefined (0/0), which in NumPy manifests as ±inf.

The fix is: (1) delete the invalid per-seed multi-seed summary files, (2) ensure `aggregate_multi_seed_results.py` is only run from the parent directory with access to all three seed subdirectories, (3) add a guard that refuses to compute statistics if n_seeds < 2 or if std=0.

### Files Involved
- `scripts/aggregate_multi_seed_results.py` — add degenerate input guards
- `experiments/results/seed_42/table_2_multi_seed_summary.csv` — DELETE (invalid)
- `experiments/results/seed_42/statistical_tests.csv` — DELETE (invalid, t=-inf)
- `experiments/results/seed_123/table_2_multi_seed_summary.csv` — DELETE (invalid)
- `experiments/results/seed_123/statistical_tests.csv` — DELETE (invalid)
- `experiments/results/seed_2024/table_2_multi_seed_summary.csv` — DELETE (invalid)
- `experiments/results/seed_2024/statistical_tests.csv` — DELETE (invalid)
- `tests/regression/test_paper_results.py` — update to use correct statistical interpretation

### Required Code Modifications

**Modification 4.1 — `scripts/aggregate_multi_seed_results.py`, add degenerate input guards**

```python
def run_statistical_tests(
    method_a_scores: list[float],
    method_b_scores: list[float],
    method_a_name: str,
    method_b_name: str,
) -> dict | None:
    """Run paired t-test and compute Cohen's d between two methods.
    
    Returns None (not a result row) if the test cannot be validly performed.
    This prevents numerically degenerate results (t=-inf, p=0) from being
    written to statistical_tests.csv.
    """
    a = np.array(method_a_scores)
    b = np.array(method_b_scores)

    # Guard 1: Minimum sample size
    if len(a) < 2:
        logger.warning(
            "Insufficient samples for statistical test",
            method_a=method_a_name,
            method_b=method_b_name,
            n_samples=len(a),
        )
        return None
    
    # Guard 2: Degenerate variance (all values identical)
    diff = a - b
    diff_std = float(np.std(diff, ddof=1))
    if diff_std < 1e-8:
        logger.warning(
            "Statistical test degenerate: std(diff) ≈ 0. "
            "This indicates all seeds produced identical results, "
            "which suggests the per-seed results are copies of a single run "
            "rather than genuinely independent experiments. "
            "Test results WILL NOT be written.",
            method_a=method_a_name,
            method_b=method_b_name,
            diff_std=diff_std,
            a_values=list(a),
            b_values=list(b),
        )
        return None
    
    # Guard 3: Values must be in valid range [0, 1] for F1
    for arr, name in [(a, method_a_name), (b, method_b_name)]:
        if np.any(arr < 0) or np.any(arr > 1):
            logger.error(
                "F1 values out of [0,1] range — data integrity issue",
                method=name,
                values=list(arr),
            )
            return None
    
    t_stat, p_value = stats.ttest_rel(a, b)
    
    # Guard 4: Check for infinite or NaN results
    if not np.isfinite(t_stat) or not np.isfinite(p_value):
        logger.error(
            "Non-finite t-statistic or p-value despite std > 0 — numerical issue",
            t_stat=float(t_stat),
            p_value=float(p_value),
            diff_std=diff_std,
        )
        return None
    
    cohens_d = float(np.mean(diff)) / (diff_std + 1e-12)

    return {
        "comparison": f"{method_a_name} vs {method_b_name}",
        "mean_diff": round(float(np.mean(diff)), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "cohens_d": round(float(cohens_d), 4),
        "significant": bool(p_value < 0.05),
        "n_seeds": len(a),
        "warning": "With n=3 seeds, statistical power is limited. p < 0.05 requires very large effect sizes." if len(a) < 5 else None,
    }
```

**Modification 4.2 — Delete invalid per-seed aggregation artifacts**

```bash
# Move invalid files to backup (do not delete without backup)
mkdir -p logs/phase4_invalid_stats_backup
cp experiments/results/seed_42/table_2_multi_seed_summary.csv logs/phase4_invalid_stats_backup/seed_42_multi_seed_summary.csv
cp experiments/results/seed_42/statistical_tests.csv logs/phase4_invalid_stats_backup/seed_42_statistical_tests.csv
# ... repeat for seed_123 and seed_2024 ...

# Remove from tracked files
git rm experiments/results/seed_42/table_2_multi_seed_summary.csv
git rm experiments/results/seed_42/statistical_tests.csv
git rm experiments/results/seed_123/table_2_multi_seed_summary.csv
git rm experiments/results/seed_123/statistical_tests.csv
git rm experiments/results/seed_2024/table_2_multi_seed_summary.csv
git rm experiments/results/seed_2024/statistical_tests.csv
```

**Modification 4.3 — Update `tests/regression/test_paper_results.py`**

Update the statistical significance test to reflect honest expectations:
```python
@pytest.mark.regression
class TestStatisticalValidity:
    def test_statistical_tests_csv_exists_and_is_valid(self):
        """Statistical tests must exist AND contain finite, valid values."""
        path = RESULTS_DIR / "statistical_tests.csv"
        if not path.exists():
            pytest.skip("Run aggregate_multi_seed_results.py first.")
        df = pd.read_csv(path)
        assert len(df) >= 1
        
        # Verify no degenerate values
        for col in ["t_statistic", "p_value", "cohens_d"]:
            if col in df.columns:
                assert not df[col].isnull().any(), f"NaN in {col}"
                assert np.isfinite(df[col].values).all(), (
                    f"Non-finite values in {col}: {df[col].values}. "
                    f"Degenerate t-test indicates seeds produced identical results."
                )
        
        # p-values must be in [0, 1]
        assert (df["p_value"] >= 0).all() and (df["p_value"] <= 1).all(), (
            f"p-values out of [0,1]: {df['p_value'].values}"
        )
    
    def test_effect_size_interpretable(self):
        """Cohen's d should be in an interpretable range (not billions)."""
        path = RESULTS_DIR / "statistical_tests.csv"
        if not path.exists():
            pytest.skip()
        df = pd.read_csv(path)
        if "cohens_d" not in df.columns:
            pytest.skip()
        assert (df["cohens_d"].abs() < 10).all(), (
            f"Cohen's d values are unreasonably large: {df['cohens_d'].values}. "
            f"Expected |d| < 10 for ML performance metrics."
        )
```

**Modification 4.4 — Add honest statistical disclaimer to README**

Add a note to the README's Table 2 section:
```markdown
> **Statistical Note**: Performance differences between methods are evaluated 
> using a paired t-test across 3 independent seeds (42, 123, 2024). With n=3 seeds, 
> statistical power is limited. Effect sizes (Cohen's d) and confidence intervals 
> are provided in `experiments/results/table_2_multi_seed_summary.csv`. 
> Differences with p > 0.05 should be interpreted as directional trends requiring 
> validation with additional seeds rather than confirmed significant differences.
```

### Validation

```bash
# Generate valid multi-seed statistics from ALL THREE seeds
python scripts/aggregate_multi_seed_results.py \
    --seeds 42 123 2024 \
    --results-dir experiments/results/

# Verify results are non-degenerate
python -c "
import pandas as pd; import numpy as np
df = pd.read_csv('experiments/results/statistical_tests.csv')
print('Statistical Tests:')
print(df.to_string(index=False))

# All t-statistics must be finite
assert np.isfinite(df['t_statistic'].values).all(), 'FAIL: Non-finite t-statistics'
# All p-values in [0, 1]
assert (df['p_value'] >= 0).all() and (df['p_value'] <= 1).all(), 'FAIL: Invalid p-values'
# Cohen's d < 10
assert (df['cohens_d'].abs() < 10).all(), 'FAIL: Degenerate Cohen d'
print('PASS: All statistical values are finite and valid')

# Report honest significance
for _, row in df.iterrows():
    sig = 'SIGNIFICANT' if row['p_value'] < 0.05 else 'NOT SIGNIFICANT'
    print(f'{row[\"comparison\"]}: p={row[\"p_value\"]:.4f} ({sig}), d={row[\"cohens_d\"]:.4f}')
"
```

### Success Criteria
- [ ] All t-statistics are finite
- [ ] All p-values in [0, 1]
- [ ] All Cohen's d values in [-10, 10]
- [ ] No per-seed multi-seed summary files exist (deleted)
- [ ] Main `statistical_tests.csv` regenerated from all 3 seeds
- [ ] Regression test for statistical validity passes

### Failure Criteria
- t-statistic is still ±inf after fix → degenerate input still reaching `ttest_rel`
- Per-seed summary CSVs still show std=0

### Git Commit Recommendation
```
git add scripts/aggregate_multi_seed_results.py \
        tests/regression/test_paper_results.py \
        README.md logs/phase4_invalid_stats_backup/
git commit -m "phase4: fix degenerate statistical tests, remove invalid per-seed aggregations

- add input guards to aggregate_multi_seed_results.py (reject std=0, n<2)
- remove per-seed multi_seed_summary and statistical_tests CSVs (invalid)
- update regression tests to verify finite t-statistics and valid p-values
- add honest statistical disclaimer to README Table 2

Fixes: CI-3 (t=-inf statistical tests)"
```

---

**PHASE COMPLETION CHECKLIST — PHASE 4**

**SUCCESS CONDITIONS:**
- statistical_tests.csv has finite values ✓
- Per-seed invalid CSVs removed ✓
- Regression test passes ✓

**FAIL CONDITIONS:**
- t-statistic still ±inf
- Any statistical value outside valid range

**NEXT PHASE ENTRY REQUIREMENTS:**
Phase 4 success criteria met. Valid statistical tests generated.

**DO NOT CONTINUE TO PHASE 5 UNTIL ALL SUCCESS CRITERIA PASS.**

---

## Phase 5 — Full Experiment Reproduction

### Objective
Re-run all experiments from scratch with the fixed codebase: FedAgent-Chain (full), Standard FedAvg (no_fairness ablation), Local Baseline, Centralized Baseline, for seeds 42, 123, and 2024. Generate all paper tables and verify internal consistency across seeds. Regenerate all figures.

### Why This Phase Matters
Every prior experiment was run with the buggy delta-accumulation FedAvg, wrong group labels in the fairness penalty, and a non-functional centralized baseline. All committed CSVs, JSONs, and blockchain logs are products of a broken codebase. They must be regenerated from scratch with the fixed code to have any scientific validity.

### Files Involved
- All of `experiments/results/` (will be overwritten with validated backups)
- All of `experiments/runs/` (generated by re-running simulations)
- `experiments/figures/` (regenerated)
- `experiments/results/table_2_model_performance.csv` through `table_7_overhead.csv`

### Pre-Phase Requirements
- Phase 1 fix verified (FedAvg convergence)
- Phase 2 fix verified (group label alignment)
- Phase 3 fix verified (centralized baseline)
- Phase 4 fix verified (statistical tests)
- Full test suite passes (no regressions)

### Tasks

**Task 5.1 — Backup Current (Broken) Results**
```bash
mkdir -p logs/phase5_broken_results_backup
cp -r experiments/results/ logs/phase5_broken_results_backup/results_before_phase5/
echo "Backup of broken results created at $(date)" > logs/phase5_broken_results_backup/README.txt
git add logs/phase5_broken_results_backup/
git commit -m "phase5: backup broken results before full re-run"
```

**Task 5.2 — Generate Fresh Synthetic Data (All Seeds)**
```bash
for SEED in 42 123 2024; do
    python scripts/generate_synthetic_data.py \
        --config configs/experiment/fedagent_chain_full.yaml \
        --seed $SEED \
        --output-dir data/synthetic/ \
        2>&1 | tee logs/phase5_data_seed${SEED}.txt
    echo "Data generation seed=$SEED exit code: $?"
done
```

Verify data was generated:
```bash
python -c "
from pathlib import Path
for node in ['saudi_arabia', 'united_states', 'china', 'europe']:
    for f in ['users.csv', 'jobs.csv', 'outcomes.csv']:
        p = Path(f'data/synthetic/{node}/{f}')
        assert p.exists(), f'Missing: {p}'
        import pandas as pd
        df = pd.read_csv(p)
        print(f'{p}: {len(df)} rows')
print('PASS: All data files present')
"
```

**Task 5.3 — Run All Configurations for Each Seed**

For each seed, run all four configurations in this order:
```bash
for SEED in 42 123 2024; do
    echo "=== Seed $SEED ==="
    
    # 1. Standard FedAvg (no_fairness ablation)
    python scripts/run_federated_simulation.py \
        --config configs/experiment/ablation/no_fairness.yaml \
        --seed $SEED --no-mlflow \
        2>&1 | tee logs/phase5_no_fairness_seed${SEED}.txt
    
    # 2. FedAgent-Chain (full, with fairness)
    python scripts/run_federated_simulation.py \
        --config configs/experiment/fedagent_chain_full.yaml \
        --seed $SEED --no-mlflow \
        2>&1 | tee logs/phase5_fedagent_seed${SEED}.txt
    
    # 3. Local Baseline
    python scripts/run_baselines.py \
        --config configs/experiment/baseline_local.yaml --seed $SEED \
        2>&1 | tee logs/phase5_local_seed${SEED}.txt
    
    # 4. Centralized Baseline
    python scripts/run_federated_simulation.py \
        --config configs/experiment/baseline_centralized.yaml \
        --seed $SEED --no-mlflow \
        2>&1 | tee logs/phase5_centralized_seed${SEED}.txt
done
```

**Task 5.4 — Convergence Diagnostics After Each Run**

After each simulation, verify convergence:
```bash
python -c "
import json, sys
from pathlib import Path

run_log = sys.argv[1]  # e.g., logs/phase5_no_fairness_seed42.txt
# Find the corresponding per_round.json
# (logic to find latest run dir)
runs = sorted(Path('experiments/runs').glob('ablation_no_fairness_seed42_*'))
latest = runs[-1]
with open(latest / 'metrics' / 'per_round.json') as f:
    history = json.load(f)

losses = [r['mean_train_loss'] for r in history]
print(f'Loss trajectory: {[round(x, 3) for x in losses]}')
assert losses[-1] < losses[0] * 3, f'FAIL: Loss diverged (final={losses[-1]:.2f}, initial={losses[0]:.2f})'
print('PASS: Loss convergence OK')
" 2>&1
```

**Task 5.5 — Generate All Evaluation Tables (Per-Seed)**

```bash
for SEED in 42 123 2024; do
    python scripts/run_evaluation.py \
        --runs-dir experiments/runs/ \
        --results-dir experiments/results/ \
        --data-dir data/synthetic \
        --seed $SEED \
        --seed-subdir \
        2>&1 | tee logs/phase5_eval_seed${SEED}.txt
done
```

After each eval, sanity check:
```bash
python -c "
import pandas as pd; import sys
seed = sys.argv[1]
df = pd.read_csv(f'experiments/results/seed_{seed}/table_2_model_performance.csv')
print(f'Seed {seed} Table 2:')
print(df[['Method','Accuracy','F1']].to_string(index=False))

# Sanity checks
for _, row in df.iterrows():
    assert 0 <= row['F1'] <= 1, f'F1 out of range for {row[\"Method\"]}: {row[\"F1\"]}'
    assert 0 <= row['Accuracy'] <= 1, f'Accuracy out of range: {row[\"Accuracy\"]}'

# Local != Centralized (Phase 3 fix must hold)
local = df[df['Method']=='Local Baseline']['F1'].values[0]
central = df[df['Method']=='Centralized']['F1'].values[0]
assert abs(local - central) > 0.005, f'FAIL: Local ({local:.4f}) == Centralized ({central:.4f})'
print('PASS: Sanity checks pass')
"
```

**Task 5.6 — Multi-Seed Aggregation**

```bash
python scripts/aggregate_multi_seed_results.py \
    --seeds 42 123 2024 \
    --results-dir experiments/results/ \
    2>&1 | tee logs/phase5_aggregate.txt

# Verify no degenerate statistics
python -c "
import pandas as pd; import numpy as np
summary = pd.read_csv('experiments/results/table_2_multi_seed_summary.csv')
print('Multi-Seed Summary:')
print(summary[['Method','F1_mean','F1_std','CI_95_lower','CI_95_upper']].to_string(index=False))

# FedAgent-Chain std must be > 0 (proving seeds are genuinely independent)
fa_row = summary[summary['Method'].str.contains('FedAgent')]
assert not fa_row.empty, 'FedAgent-Chain row missing'
fa_std = float(fa_row['F1_std'].values[0])
assert fa_std > 0.001, f'FAIL: FedAgent F1_std={fa_std} ≈ 0, seeds are not independent'

stats = pd.read_csv('experiments/results/statistical_tests.csv')
print()
print('Statistical Tests:')
print(stats.to_string(index=False))
assert np.isfinite(stats['t_statistic'].values).all(), 'FAIL: Non-finite t-statistics'
print('PASS: Statistics valid')
"
```

**Task 5.7 — Regenerate Figures**

```bash
python scripts/generate_figures.py \
    --results-dir experiments/results/ \
    --runs-dir experiments/runs/ \
    --output-dir experiments/figures/ \
    2>&1 | tee logs/phase5_figures.txt

# Verify figure files exist
python -c "
from pathlib import Path
figures = ['fl_convergence.pdf', 'node_f1_scores.pdf', 'fairness_disparity.pdf']
for fig in figures:
    p = Path(f'experiments/figures/{fig}')
    assert p.exists(), f'MISSING FIGURE: {fig}'
    assert p.stat().st_size > 1000, f'Figure too small (likely empty): {fig}'
    print(f'OK: {fig} ({p.stat().st_size} bytes)')
"
```

**Task 5.8 — Full Regression Test Suite**

```bash
# Update regression test reference values with new results
python -c "
import pandas as pd
df = pd.read_csv('experiments/results/table_2_multi_seed_summary.csv')
fa_f1 = float(df[df['Method'].str.contains('FedAgent')]['F1_mean'].values[0])
print(f'NEW REFERENCE: FedAgent-Chain F1_mean = {fa_f1:.4f}')
print('Update REFERENCE dict in tests/regression/test_paper_results.py')
"
# Manually update tests/regression/test_paper_results.py REFERENCE dict with new values
# Then run:
pytest tests/regression/ -v --timeout=600 2>&1 | tee logs/phase5_regression.txt
```

### Expected Outputs
- `experiments/results/seed_{42,123,2024}/table_2_model_performance.csv` — per-seed results
- `experiments/results/table_2_model_performance.csv` — primary result table
- `experiments/results/table_2_multi_seed_summary.csv` — multi-seed aggregation with F1_std > 0
- `experiments/results/table_3_fairness_results.csv` — fairness disparity (should show improvement)
- `experiments/results/statistical_tests.csv` — valid finite statistics
- `experiments/figures/{fl_convergence,node_f1_scores,fairness_disparity}.pdf`

### Success Criteria
- [ ] All 4 configurations run for all 3 seeds (12 simulation runs total)
- [ ] All 20-round runs show non-diverging loss (final loss < 3.0)
- [ ] Local Baseline ≠ Centralized (F1 difference > 0.005)
- [ ] FedAgent-Chain F1_std > 0.001 (seeds genuinely independent)
- [ ] All statistical test values are finite
- [ ] All 3 figures generated and non-empty
- [ ] Regression tests pass with updated reference values

### Failure Criteria
- Any simulation run produces diverging loss
- Local == Centralized outputs (Phase 3 fix failed to propagate)
- F1_std == 0 after aggregation (seeds not truly independent)

### Git Commit Recommendation
```
git add experiments/results/ experiments/figures/ \
        tests/regression/test_paper_results.py \
        logs/phase5_*.txt
git commit -m "phase5: full experiment reproduction with fixed codebase

- all 12 runs (4 configs × 3 seeds) completed with fixed FedAvg
- FedAvg loss convergence verified (non-diverging)
- centralized != local baseline verified
- multi-seed aggregation generates valid statistics
- paper figures regenerated from real simulation outputs

Resolves all CI issues. See REPAIR_LOG.md for detailed comparison."
```

---

**PHASE COMPLETION CHECKLIST — PHASE 5**

**SUCCESS CONDITIONS:**
- 12 simulation runs completed ✓
- Non-diverging loss across all runs ✓
- Valid multi-seed statistics ✓
- All figures generated ✓
- Regression tests pass ✓

**FAIL CONDITIONS:**
- Any run crashes or diverges
- Centralized still equals Local
- F1_std still 0

**NEXT PHASE ENTRY REQUIREMENTS:**
All 12 runs completed, all tables regenerated, regression tests pass.

**DO NOT CONTINUE TO PHASE 6 UNTIL ALL SUCCESS CRITERIA PASS.**

---

## Phase 6 — Fairness-Accuracy Tradeoff Study (Required Missing Experiment)

### Objective
Conduct the λ_fairness sweep experiment to characterize the tradeoff between F1 accuracy and fairness disparity D_fair. This experiment is required to: (1) justify the choice of λ=0.5 in the paper, (2) provide the Pareto frontier evidence that the fairness penalty creates a genuine tradeoff, and (3) determine whether any λ value achieves both better F1 AND better fairness than Standard FedAvg.

### Why This Phase Matters
Without this experiment, the paper cannot justify its choice of λ=0.5. Currently the table shows FedAgent-Chain achieves lower F1 than Standard FedAvg for the same or worse fairness (before fixes). This experiment either (a) shows that a different λ value achieves better tradeoffs — providing a scientifically valid contribution — or (b) shows that no λ value achieves the claimed benefits — requiring fundamental redesign of the fairness mechanism.

### Tasks

**Task 6.1 — Create Lambda Sweep Configurations**

Create `configs/experiment/lambda_sweep/` with the following configs (one per λ value):
```yaml
# configs/experiment/lambda_sweep/lambda_0.00.yaml
experiment:
  name: lambda_sweep_0.00
  description: "Lambda fairness sweep: lambda=0.00 (Standard FedAvg)"
  variant: lambda_sweep
seed: 42
federated:
  n_rounds: 20
  min_clients: 4
  local_epochs: 3
  batch_size: 64
  learning_rate: 0.0001
privacy:
  enabled: true
  clipping_threshold: 1.0
  noise_multiplier: 0.1
fairness:
  enabled: true
  lambda_fairness: 0.00   # varies per config
  protected_groups:
    - disability_category
    - language_primary
    - preferred_work_mode
    - node_id
blockchain:
  enabled: false   # disabled for speed in sweep
data:
  n_users_per_node: 2500
  n_jobs_per_node: 1250
  n_pairs_per_node: 12500
```

Create configs for λ ∈ {0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00}.

**Task 6.2 — Run Lambda Sweep Script**

Create `scripts/run_lambda_sweep.py`:
```python
#!/usr/bin/env python3
"""Lambda fairness sweep for FedAgent-Chain tradeoff analysis.

Runs FedAgent-Chain for each lambda value and collects F1 and D_fair(disability).
"""
import subprocess, sys
from pathlib import Path

LAMBDAS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00]
SEED = 42

for lam in LAMBDAS:
    config_path = Path(f"configs/experiment/lambda_sweep/lambda_{lam:.2f}.yaml")
    assert config_path.exists(), f"Config missing: {config_path}"
    
    print(f"\n=== Lambda = {lam:.2f} ===")
    result = subprocess.run([
        sys.executable, "scripts/run_federated_simulation.py",
        "--config", str(config_path),
        "--seed", str(SEED),
        "--no-mlflow",
    ], check=False)
    
    if result.returncode != 0:
        print(f"FAIL: lambda={lam:.2f} simulation failed with code {result.returncode}")
    else:
        print(f"PASS: lambda={lam:.2f} simulation complete")
```

**Task 6.3 — Aggregate and Plot Tradeoff Curve**

Create `scripts/generate_lambda_tradeoff_plot.py`:
```python
"""Aggregate lambda sweep results and generate F1 vs D_fair tradeoff plot."""
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Collect metrics for each lambda
results = []
for lam_config in sorted(Path("configs/experiment/lambda_sweep").glob("*.yaml")):
    lam = float(lam_config.stem.split("_")[-1])
    # Find the corresponding run directory
    runs = sorted(Path("experiments/runs").glob(f"lambda_sweep_{lam:.2f}_seed42_*"))
    if not runs:
        print(f"WARNING: No run found for lambda={lam:.2f}")
        continue
    latest = runs[-1]
    
    # Load per-round metrics
    final_json = latest / "metrics" / "final.json"
    if not final_json.exists():
        print(f"WARNING: No final.json for lambda={lam:.2f}")
        continue
    
    with open(final_json) as f:
        final = json.load(f)
    
    best_f1 = final.get("best_f1", float("nan"))
    mean_disparity = final.get("final_round_metrics", {}).get(
        "mean_fairness_disparity_disability", float("nan")
    )
    
    results.append({
        "lambda": lam,
        "best_f1": best_f1,
        "d_fair_disability": mean_disparity,
    })

df = pd.DataFrame(results).sort_values("lambda")
df.to_csv("experiments/results/lambda_tradeoff.csv", index=False)
print(df.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(df["d_fair_disability"], df["best_f1"], 
                     c=df["lambda"], cmap="viridis", s=100, zorder=5)
for _, row in df.iterrows():
    ax.annotate(f"λ={row['lambda']:.2f}", 
                (row["d_fair_disability"], row["best_f1"]),
                textcoords="offset points", xytext=(5, 5))
plt.colorbar(scatter, label="λ_fairness")
ax.set_xlabel(r"D_fair (disability) — lower is fairer", fontsize=12)
ax.set_ylabel("Best F1 Score", fontsize=12)
ax.set_title("Fairness-Accuracy Tradeoff: λ Sweep", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
fig.savefig("experiments/figures/lambda_tradeoff.pdf", dpi=300, bbox_inches="tight")
print("Figure saved: experiments/figures/lambda_tradeoff.pdf")
```

**Task 6.4 — Europe Node Failure Diagnosis**

Run Europe-only local training to determine if the near-zero recall is a data distribution issue or FedAvg artifact:
```bash
# Create a minimal Europe-only config
cat > configs/experiment/europe_local_diagnosis.yaml << EOF
experiment:
  name: europe_local_diagnosis
seed: 42
federated:
  n_rounds: 1
  local_epochs: 50
  batch_size: 64
  learning_rate: 0.001
  min_clients: 1
privacy:
  enabled: false
fairness:
  enabled: false
blockchain:
  enabled: false
data:
  nodes: [europe]
  n_users_per_node: 2500
  n_jobs_per_node: 1250
  n_pairs_per_node: 12500
EOF

python scripts/run_federated_simulation.py \
    --config configs/experiment/europe_local_diagnosis.yaml \
    --seed 42 --no-mlflow \
    2>&1 | tee logs/phase6_europe_diagnosis.txt

# Report findings
python -c "
import json
from pathlib import Path
runs = sorted(Path('experiments/runs').glob('europe_local_diagnosis_*'))
with open(runs[-1] / 'metrics' / 'per_round.json') as f:
    history = json.load(f)
final = history[-1]
print('Europe-only local training (50 epochs):')
print(f'  F1: {final[\"per_node\"][\"europe\"][\"f1\"]:.4f}')
print(f'  Recall: {final[\"per_node\"][\"europe\"][\"recall\"]:.4f}')
print(f'  Accuracy: {final[\"per_node\"][\"europe\"][\"accuracy\"]:.4f}')
recall = final['per_node']['europe']['recall']
if recall > 0.5:
    print('FINDING: Europe succeeds with local training → FedAvg aggregation is destroying Europe model')
    print('RECOMMENDATION: Consider FedProx or per-node learning rate adaptation')
else:
    print('FINDING: Europe fails even locally → Data distribution issue')
    print('RECOMMENDATION: Check Europe disability/language distribution, consider SMOTE or resampling')
" 2>&1 | tee -a logs/phase6_europe_diagnosis.txt
```

Document findings in `REPAIR_LOG.md`.

### Validation

After the sweep:
- Verify the Pareto frontier exists (some λ values have better D_fair at cost of F1)
- Verify λ=0 approximately recovers Standard FedAvg results
- Document the optimal λ that minimizes combined loss (e.g., `F1_loss + D_fair`)
- If no λ achieves better fairness than Standard FedAvg, document this as a limitation requiring future work

### Success Criteria
- [ ] Lambda sweep completed for all 8 λ values
- [ ] `experiments/results/lambda_tradeoff.csv` generated
- [ ] `experiments/figures/lambda_tradeoff.pdf` generated
- [ ] Europe node diagnosis completed and findings documented in REPAIR_LOG.md
- [ ] Optimal λ identified and justified in REPAIR_LOG.md

### Failure Criteria
- No λ value achieves better fairness AND better F1 simultaneously (expected — document as tradeoff)
- Lambda sweep simulations diverge (indicates Phase 1 fix is incomplete)

### Git Commit Recommendation
```
git add configs/experiment/lambda_sweep/ scripts/run_lambda_sweep.py \
        scripts/generate_lambda_tradeoff_plot.py \
        experiments/results/lambda_tradeoff.csv \
        experiments/figures/lambda_tradeoff.pdf \
        logs/phase6_europe_diagnosis.txt
git commit -m "phase6: add lambda fairness sweep and Europe node diagnosis

- sweep lambda in {0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00}
- generate F1 vs D_fair Pareto frontier plot
- diagnose Europe node failure mode (local vs FedAvg)
- document optimal lambda and tradeoff characterization

Addresses: missing experiment ME-1 (fairness-accuracy tradeoff)"
```

---

**PHASE COMPLETION CHECKLIST — PHASE 6**

**SUCCESS CONDITIONS:**
- All 8 lambda sweep runs complete ✓
- Tradeoff CSV and plot generated ✓
- Europe diagnosis documented ✓

**FAIL CONDITIONS:**
- Lambda sweep simulations diverge (implies Phase 1 incomplete)

**NEXT PHASE ENTRY REQUIREMENTS:**
Lambda sweep complete. Europe diagnosis documented.

**DO NOT CONTINUE TO PHASE 7 UNTIL ALL SUCCESS CRITERIA PASS.**

---

## Phase 7 — Repository Cleanup and Hardening

### Objective
Remove all identified development artifacts, dead code, and invalid committed results. Harden the codebase with defensive assertions, CI improvements, and reproducibility scripts. Ensure the repository is clean enough for public submission.

### Tasks

**Task 7.1 — Remove Dead Code and Development Artifacts**

```bash
# Remove development debug files
git rm debug_eval.py
git rm per_round_debug.json

# Remove temp results directory
git rm -r experiments/results_temp/

# Remove already-backed-up invalid stat files (done in Phase 4)
# Verify they are already removed
git status experiments/results/seed_42/statistical_tests.csv  # should show 'deleted'
```

**Task 7.2 — Fix Dead Code in `_find_best_checkpoint`**
Verify this was already done in Phase 3. Run a specific test:
```python
def test_find_best_checkpoint_fallback_reachable():
    """Verify the fallback code in _find_best_checkpoint is reachable
    by testing with a run_dir that has per_round.json but no matching checkpoint."""
    import tempfile
    from pathlib import Path
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        ckpt_dir = run_dir / "checkpoints"
        metrics_dir = run_dir / "metrics"
        ckpt_dir.mkdir()
        metrics_dir.mkdir()
        
        # Create per_round.json saying best round is 5
        history = [{"round": i, "mean_f1": 0.5 + i * 0.01} for i in range(1, 6)]
        with open(metrics_dir / "per_round.json", "w") as f:
            json.dump(history, f)
        
        # Create checkpoints for rounds 1, 2, 3 only (NOT 5, which is "best")
        for rnd in [1, 2, 3]:
            (ckpt_dir / f"global_model_round_{rnd:03d}.pt").touch()
        
        # Also create final_model.pt as fallback
        (ckpt_dir / "final_model.pt").touch()
        
        # Should return round 3 (nearest below best round 5)
        from scripts.run_evaluation import _find_best_checkpoint
        result = _find_best_checkpoint(run_dir)
        assert result.name == "global_model_round_003.pt", (
            f"Expected round 3 fallback, got {result.name}"
        )
```

**Task 7.3 — Add Loss Explosion Detector as Permanent CI Check**

Add to `.github/workflows/ci.yml`:
```yaml
- name: Verify no diverging loss in smoke test
  run: |
    python -c "
    import json
    from pathlib import Path
    runs = sorted(Path('experiments/runs').glob('fedagent_chain_full_seed42_*'))
    if not runs:
        print('No runs found - skipping divergence check')
        exit(0)
    latest = runs[-1]
    per_round = latest / 'metrics' / 'per_round.json'
    if not per_round.exists():
        print('No per_round.json - skipping')
        exit(0)
    with open(per_round) as f:
        history = json.load(f)
    if len(history) < 2:
        exit(0)
    losses = [r['mean_train_loss'] for r in history]
    ratio = losses[-1] / (losses[0] + 1e-8)
    assert ratio < 5.0, f'Loss diverged: round1={losses[0]:.3f}, final={losses[-1]:.3f}, ratio={ratio:.1f}'
    print(f'Loss convergence OK: {losses[0]:.3f} -> {losses[-1]:.3f}')
    "
```

**Task 7.4 — Add Artifact Integrity Check Script**

Create `scripts/verify_artifact_integrity.py`:
```python
#!/usr/bin/env python3
"""Verify that all committed result artifacts are internally consistent.

Checks:
1. Per-seed tables exist for all 3 seeds
2. Multi-seed summary std > 0 (seeds are independent)  
3. Statistical tests have finite values
4. Centralized != Local Baseline
5. FedAgent-Chain fairness improvement exists for at least 2/4 attributes
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RESULTS = Path("experiments/results")
errors = []
warnings = []

# Check 1: Per-seed tables exist
for seed in [42, 123, 2024]:
    for table in ["table_2_model_performance.csv", "table_3_fairness_results.csv"]:
        p = RESULTS / f"seed_{seed}" / table
        if not p.exists():
            errors.append(f"MISSING: {p}")

# Check 2: Multi-seed std > 0
summary = RESULTS / "table_2_multi_seed_summary.csv"
if summary.exists():
    df = pd.read_csv(summary)
    fa_row = df[df["Method"].str.contains("FedAgent", na=False)]
    if not fa_row.empty and float(fa_row["F1_std"].values[0]) < 0.001:
        errors.append("F1_std ≈ 0 in multi-seed summary — seeds are not independent")

# Check 3: Statistical tests finite
stats_path = RESULTS / "statistical_tests.csv"
if stats_path.exists():
    stats_df = pd.read_csv(stats_path)
    if not np.isfinite(stats_df["t_statistic"].values).all():
        errors.append("Non-finite t-statistics in statistical_tests.csv")

# Check 4: Centralized != Local
t2 = RESULTS / "table_2_model_performance.csv"
if t2.exists():
    df = pd.read_csv(t2)
    local = df[df["Method"] == "Local Baseline"]["F1"].values
    central = df[df["Method"] == "Centralized"]["F1"].values
    if len(local) > 0 and len(central) > 0:
        if abs(float(local[0]) - float(central[0])) < 0.005:
            errors.append(f"Centralized F1 ({float(central[0]):.4f}) == Local Baseline F1 ({float(local[0]):.4f})")

# Report
print("=== ARTIFACT INTEGRITY CHECK ===")
if errors:
    print(f"\n{len(errors)} ERRORS:")
    for e in errors:
        print(f"  ERROR: {e}")
    sys.exit(1)
else:
    print("\nPASS: All artifact integrity checks passed")
```

**Task 7.5 — Update CHANGELOG.md and REPAIR_LOG.md**

Document all changes made in phases 1-6 in `CHANGELOG.md` with version `1.1.0-repaired`.

**Task 7.6 — Add Checkpoint Integrity Validator**

Create `scripts/validate_checkpoints.py`:
```python
#!/usr/bin/env python3
"""Validate all model checkpoints for integrity."""
import torch
import numpy as np
from pathlib import Path
import sys

def validate_checkpoint(ckpt_path: Path) -> list[str]:
    """Return list of errors, empty if checkpoint is valid."""
    errors = []
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        return [f"Cannot load checkpoint: {e}"]
    
    for name, tensor in state_dict.items():
        arr = tensor.numpy()
        if np.any(np.isnan(arr)):
            errors.append(f"NaN in parameter '{name}'")
        if np.any(np.isinf(arr)):
            errors.append(f"Inf in parameter '{name}'")
        norm = float(np.linalg.norm(arr))
        if norm > 10000:
            errors.append(f"Suspiciously large weight norm in '{name}': {norm:.2f}")
    return errors

# Validate all checkpoints in experiments/runs/
for ckpt in sorted(Path("experiments/runs").rglob("*.pt")):
    errs = validate_checkpoint(ckpt)
    if errs:
        print(f"FAIL: {ckpt}")
        for e in errs:
            print(f"  {e}")
    else:
        print(f"OK: {ckpt.name}")
```

### Success Criteria
- [ ] `debug_eval.py` removed from repository
- [ ] `per_round_debug.json` removed
- [ ] `experiments/results_temp/` removed
- [ ] Dead code after `return` in `_find_best_checkpoint` removed (Phase 3)
- [ ] `scripts/verify_artifact_integrity.py` passes with no errors
- [ ] All checkpoints validated (no NaN, no Inf, weight norms < 10000)
- [ ] CI workflow updated with loss divergence check
- [ ] CHANGELOG.md updated

### Git Commit Recommendation
```
git add . --all
git commit -m "phase7: cleanup, hardening, CI improvements

- remove debug_eval.py, per_round_debug.json, experiments/results_temp/
- add artifact integrity check script
- add checkpoint validator
- add CI loss divergence check
- update CHANGELOG.md with all phase 1-6 changes"
```

---

**PHASE COMPLETION CHECKLIST — PHASE 7**

**SUCCESS CONDITIONS:**
- All artifacts removed ✓
- Integrity check passes ✓
- Checkpoints valid ✓

**FAIL CONDITIONS:**
- Integrity check reports errors

**NEXT PHASE ENTRY REQUIREMENTS:**
verify_artifact_integrity.py passes with exit code 0.

**DO NOT CONTINUE TO PHASE 8 UNTIL ALL SUCCESS CRITERIA PASS.**

---

## Phase 8 — Paper Result Regeneration and Table Verification

### Objective
Regenerate all paper tables from scratch. Cross-reference every number in the paper's README against the generated CSVs. Update the README tables with new values. Generate a comprehensive result comparison report showing before-repair vs after-repair values for every metric.

### Tasks

**Task 8.1 — Generate Final Paper Tables**

```bash
# Generate all tables from final run results
python scripts/run_evaluation.py \
    --runs-dir experiments/runs/ \
    --results-dir experiments/results/ \
    --data-dir data/synthetic \
    --seed 42

python scripts/generate_tables.py --results-dir experiments/results/
python scripts/generate_figures.py \
    --results-dir experiments/results/ \
    --runs-dir experiments/runs/ \
    --output-dir experiments/figures/
python scripts/export_blockchain_audit.py --output-dir experiments/results/
python scripts/aggregate_multi_seed_results.py --seeds 42 123 2024 --results-dir experiments/results/
```

**Task 8.2 — Before-vs-After Comparison Report**

Create `experiments/results/repair_comparison_report.md` documenting every metric change:
```markdown
# FedAgent-Chain Repair: Before vs After Comparison

## Table 2: Model Performance (F1)
| Method | Before (Broken) | After (Fixed) | Change | Note |
|--------|-----------------|---------------|--------|------|
| FedAgent-Chain | 0.6374 | [NEW] | [DELTA] | |
| Standard FedAvg | 0.6762 | [NEW] | [DELTA] | FedAvg was diverging; best ckpt only |
| Local Baseline | 0.6782 | [NEW] | [DELTA] | |
| Centralized | 0.6782 | [NEW] | [DELTA] | Was identical to local, now correct |

## Table 3: Fairness Disparity D_fair (disability)
| Method | Before | After | Change |
|--------|--------|-------|--------|
| FedAgent-Chain | 0.0729 | [NEW] | |
| Standard FedAvg | 0.0354 | [NEW] | |
| Reduction | -105.9% (WRONG: increase) | [NEW] | |
```

**Task 8.3 — Update README.md Tables**

Replace all hardcoded metric tables in README.md with the regenerated values from:
- `experiments/results/table_2_model_performance.csv`
- `experiments/results/table_3_fairness_results.csv`
- `experiments/results/table_4_blockchain_results.csv`
- `experiments/results/table_7_overhead.csv`

**Task 8.4 — Generate Final Checksums**

```bash
find experiments/results/ -name "*.csv" -o -name "*.json" | sort | xargs sha256sum > logs/phase8_artifact_checksums_final.txt
echo "Final checksum registry created at $(date)" >> logs/phase8_artifact_checksums_final.txt

# Compare with Phase 0 checksums
diff logs/phase0_artifact_checksums_before.txt logs/phase8_artifact_checksums_final.txt | head -100
```

**Task 8.5 — Run `verify_submission_readiness.py`**

```bash
python scripts/verify_submission_readiness.py 2>&1 | tee logs/phase8_submission_check.txt

# This script must pass without errors
python -c "
import sys
with open('logs/phase8_submission_check.txt') as f:
    content = f.read()
if 'ERROR' in content:
    print('FAIL: Submission readiness check has errors')
    sys.exit(1)
else:
    print('PASS: Submission readiness check passed')
"
```

### Success Criteria
- [ ] `verify_submission_readiness.py` passes with no errors
- [ ] Before-vs-after comparison report committed
- [ ] README.md updated with new metric values
- [ ] All figures are PDFs with file size > 5KB (not empty)
- [ ] Final checksums committed

### Git Commit Recommendation
```
git add experiments/results/ experiments/figures/ README.md \
        logs/phase8_*.txt
git commit -m "phase8: regenerate all paper tables and figures from fixed code

- all tables regenerated from clean experiments with fixed FedAvg + fairness
- README updated with corrected metric values
- before/after comparison report committed
- submission readiness check: PASS"
```

---

**PHASE COMPLETION CHECKLIST — PHASE 8**

**SUCCESS CONDITIONS:**
- verify_submission_readiness.py exits 0 ✓
- README updated ✓
- Checksums committed ✓

**FAIL CONDITIONS:**
- Submission readiness has errors

**NEXT PHASE ENTRY REQUIREMENTS:**
Submission readiness passes.

**DO NOT CONTINUE TO PHASE 9 UNTIL ALL SUCCESS CRITERIA PASS.**

---

## Phase 9 — Final Verification and Submission Readiness

### Objective
Perform final end-to-end verification: clone the repository fresh into a temporary directory, install from scratch, run the full pipeline, and verify the outputs match the committed artifacts. Confirm the paper can be reproduced by a third party.

### Tasks

**Task 9.1 — Fresh Clone Verification**
```bash
cd /tmp
git clone https://github.com/aliakarma/fedagent-chain fedagent-chain-verify
cd fedagent-chain-verify
conda env create -f environment.yml
conda activate fedagent-chain
pip install -e .
pytest tests/unit/ -v --tb=short
pytest tests/integration/ -v -m integration --timeout=120
```

**Task 9.2 — Minimal Reproducibility Run**
```bash
cd /tmp/fedagent-chain-verify
make reproduce  # Should run full pipeline

# OR manually:
python scripts/generate_synthetic_data.py --config configs/experiment/fedagent_chain_full.yaml --seed 42
python scripts/run_federated_simulation.py --config configs/experiment/fedagent_chain_full.yaml --seed 42 --no-mlflow
python scripts/run_evaluation.py --seed 42
python scripts/aggregate_multi_seed_results.py --seeds 42 --results-dir experiments/results/
```

**Task 9.3 — Numerical Consistency Check**
```bash
python -c "
import pandas as pd
from pathlib import Path
# Compare fresh-run table_2 against committed table_2 within tolerance
fresh = pd.read_csv('experiments/results/table_2_model_performance.csv')
committed = pd.read_csv('/tmp/fedagent-chain-verify/experiments/results/table_2_model_performance.csv')
for col in ['F1', 'Accuracy']:
    for method in fresh['Method']:
        a = float(fresh[fresh['Method']==method][col].values[0])
        b = float(committed[committed['Method']==method][col].values[0])
        assert abs(a - b) < 0.015, f'Reproducibility check FAILED for {method} {col}: {a:.4f} vs {b:.4f}'
print('PASS: Fresh run reproduces committed results within tolerance=0.015')
"
```

**Task 9.4 — Final Paper Claims Audit**

Manually verify each claim in the abstract and contribution list against the new results:
1. "Reduces disability disparity" — verify D_fair(disability) FedAgent-Chain < Standard FedAvg in new table_3
2. "Blockchain hash completeness 100%" — verify table_4
3. "Agentic AI coverage" — verify table_5
4. Statistical significance claims — verify against statistical_tests.csv

For any claim that is NOT supported by the new results, update the claim in the paper or in the README.

**Task 9.5 — Final Commit and Tag**

```bash
git add .
git commit -m "phase9: final verification, fresh clone test passes

- fresh clone reproducibility verified within tolerance=0.015
- all paper claims verified against new results
- REPAIR_LOG.md finalized with all changes documented"

git tag -a v1.1.0-repaired -m "Version 1.1.0 with all Phase 0-8 repairs applied"
git push origin main v1.1.0-repaired
```

### Success Criteria
- [ ] Fresh clone pip install succeeds
- [ ] Fresh clone unit tests pass
- [ ] Fresh clone single-seed reproduction within F1 tolerance 0.015 of committed results
- [ ] All paper claims verified against new result tables
- [ ] `REPAIR_LOG.md` fully complete
- [ ] Git tag `v1.1.0-repaired` pushed

---

**PHASE COMPLETION CHECKLIST — PHASE 9**

**SUCCESS CONDITIONS:**
- Fresh clone reproduces within tolerance ✓
- All paper claims verified ✓
- Git tag pushed ✓

**FAIL CONDITIONS:**
- Fresh clone cannot install dependencies
- Reproduction error > 0.015 F1
- Any paper claim unsupported by results

**NEXT PHASE ENTRY REQUIREMENTS:**
This is the final phase. All success criteria must pass before declaring the repository repaired.

**DO NOT DECLARE REPAIR COMPLETE UNTIL ALL PHASE 9 SUCCESS CRITERIA PASS.**

---

## Master Completion Protocol

After each completed phase, you MUST provide the following structured summary before proceeding:

```
=== PHASE [N] COMPLETION SUMMARY ===

1. WHAT WAS FIXED:
   - [List each specific change made, file by file]
   - [Include before/after code snippets for critical fixes]

2. REMAINING RISKS:
   - [List any risks that were identified but not fully resolved]
   - [Include confidence level for each fix]

3. VALIDATION EVIDENCE:
   - [Paste test output showing pass/fail results]
   - [Include key metric values (loss curves, F1 scores)]
   - [Include any diagnostic outputs]

4. MODIFIED FILES:
   - [Complete list of files changed with description of change]

5. NEWLY ADDED TESTS:
   - [Complete list of new test functions and their purpose]
   - [Pass/fail status for each]

6. REPOSITORY INTEGRITY:
   - [State whether existing tests still pass]
   - [State whether artifacts are consistent]
   - [State whether checksums have been updated]

7. SAFE TO PROCEED: YES / NO
   - If NO: explain exactly what must happen before proceeding
```

This summary is **mandatory** and **non-negotiable**. Any phase summary that omits a section or provides vague answers (e.g., "tests pass" without showing output, or "metrics improved" without showing values) must be treated as an incomplete phase summary and the phase must be re-executed or the summary re-generated with full evidence.

---

*End of Master Remediation Plan. This document represents the complete engineering specification for repairing the FedAgent-Chain repository. All phases must be executed in order. All success criteria must be verified before phase transitions. All statistical claims must be supported by finite, valid numerical evidence. No result may be assumed without measurement.*