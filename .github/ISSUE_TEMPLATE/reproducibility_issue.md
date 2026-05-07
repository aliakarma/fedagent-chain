---
name: Reproducibility Issue
about: Report that paper results cannot be reproduced from this repository
title: "[REPRODUCIBILITY] "
labels: reproducibility
assignees: ""
---

## Which Result Cannot Be Reproduced
Specify the table, figure, or metric (e.g., "Table 2 FedAgent-Chain F1 score").

## Expected Value (from paper)
F1 = 0.832 (for example)

## Observed Value
F1 = X.XXX

## Deviation
|observed - expected| = X.XXX (tolerance in regression tests is ±0.005)

## Reproduction Steps Followed
- [ ] Ran `make reproduce` or the individual scripts
- [ ] Used seed=42 as specified
- [ ] Generated synthetic data with the provided script
- [ ] Used the config file `configs/experiment/fedagent_chain_full.yaml`

## Command Used
```bash
python scripts/run_federated_simulation.py --config ... --seed 42
```

## Environment
- OS:
- Python:
- PyTorch:
- Hardware (CPU/GPU):

## Additional Notes
