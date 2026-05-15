from pathlib import Path
lambdas = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00]
template = """# Lambda fairness sweep: lambda={lam:.2f}
defaults:
  - /default

experiment:
  name: lambda_sweep_{lam:.2f}
  description: "Lambda fairness sweep: lambda={lam:.2f}"
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
  lambda_fairness: {lam:.2f}
  protected_groups:
    - disability_category
    - language_primary
    - preferred_work_mode
    - node_id

blockchain:
  enabled: false

data:
  n_users_per_node: 2500
  n_jobs_per_node: 1250
  n_pairs_per_node: 12500
"""
for lam in lambdas:
    p = Path(f"configs/experiment/lambda_sweep/lambda_{lam:.2f}.yaml")
    p.write_text(template.format(lam=lam))
    print(f"Created: {p}")
