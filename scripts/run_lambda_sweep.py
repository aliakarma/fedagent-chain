#!/usr/bin/env python3
"""Lambda fairness sweep for FedAgent-Chain tradeoff analysis.

Runs FedAgent-Chain for each lambda value and collects F1 and D_fair(disability).
"""
import subprocess
import sys
from pathlib import Path

LAMBDAS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00]
SEED = 42

for lam in LAMBDAS:
    config_path = Path(f"configs/experiment/lambda_sweep/lambda_{lam:.2f}.yaml")
    assert config_path.exists(), f"Config missing: {config_path}"
    
    print(f"\n=== Lambda = {lam:.2f} ===")
    # Add PYTHONPATH to include src and root, preserving existing environment
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd()}/src;{Path.cwd()}"
    
    result = subprocess.run([
        sys.executable, "scripts/run_federated_simulation.py",
        "--config", str(config_path),
        "--seed", str(SEED),
        "--no-mlflow",
    ], check=False, env=env)
    
    if result.returncode != 0:
        print(f"FAIL: lambda={lam:.2f} simulation failed with code {result.returncode}")
    else:
        print(f"PASS: lambda={lam:.2f} simulation complete")
