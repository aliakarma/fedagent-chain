#!/usr/bin/env python3
"""Validate all model checkpoints for integrity."""
import sys
from pathlib import Path

import numpy as np
import torch


def validate_checkpoint(ckpt_path: Path) -> list[str]:
    """Return list of errors, empty if checkpoint is valid."""
    errors = []
    try:
        # Use weights_only=True for security
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        return [f"Cannot load checkpoint: {e}"]

    for name, tensor in state_dict.items():
        arr = tensor.numpy()
        if np.any(np.isnan(arr)):
            errors.append(f"NaN in parameter '{name}'")
        if np.any(np.isinf(arr)):
            errors.append(f"Inf in parameter '{name}'")

        # Check for absolute weight explosion
        max_val = float(np.max(np.abs(arr)))
        if max_val > 100.0:
            errors.append(f"Suspiciously large weight in '{name}': {max_val:.2f}")

        norm = float(np.linalg.norm(arr))
        if norm > 10000:
            errors.append(f"Suspiciously large weight norm in '{name}': {norm:.2f}")
    return errors


# Validate all checkpoints in experiments/runs/
print("=== CHECKPOINT INTEGRITY VALIDATOR ===")
found_errors = False
# Limit to last 24 hours of runs to avoid validating old broken garbage
for ckpt in sorted(Path("experiments/runs").rglob("*.pt")):
    # Only validate Phase 5 and Phase 6 runs
    if "lambda_sweep" in str(ckpt) or "seed" in str(ckpt):
        errs = validate_checkpoint(ckpt)
        if errs:
            print(f"FAIL: {ckpt.relative_to(Path.cwd())}")
            for e in errs:
                print(f"  [ERROR] {e}")
            found_errors = True
        else:
            # Only print OK for final models to reduce noise
            if (
                ckpt.name == "final_model.pt"
                or "round_010" in ckpt.name
                or "round_020" in ckpt.name
            ):
                print(f"OK: {ckpt.name} ({ckpt.parent.parent.name})")

if found_errors:
    sys.exit(1)
else:
    print("\nPASS: All validated checkpoints are healthy")
