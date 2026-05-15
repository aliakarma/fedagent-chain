"""Aggregate lambda sweep results and generate F1 vs D_fair tradeoff plot."""
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Collect metrics for each lambda
results = []
for lam_config in sorted(Path("configs/experiment/lambda_sweep").glob("*.yaml")):
    try:
        lam_str = lam_config.stem.split("_")[-1]
        lam = float(lam_str)
    except (IndexError, ValueError):
        print(f"Skipping invalid config: {lam_config}")
        continue
        
    # Find the corresponding run directory
    # Format: lambda_sweep_{lam:.2f}_seed42_*
    pattern = f"lambda_sweep_{lam:.2f}_seed42_*"
    runs = sorted(Path("experiments/runs").glob(pattern))
    if not runs:
        print(f"WARNING: No run found for lambda={lam:.2f}")
        continue
    latest = runs[-1]
    
    # Load per-round metrics
    final_json = latest / "metrics" / "final.json"
    if not final_json.exists():
        # Fallback to per_round.json latest entry
        per_round_json = latest / "metrics" / "per_round.json"
        if not per_round_json.exists():
            print(f"WARNING: No metrics for lambda={lam:.2f}")
            continue
        with open(per_round_json) as f:
            history = json.load(f)
        final_metrics = history[-1]
        best_f1 = max(r.get("avg_f1", 0) for r in history)
    else:
        with open(final_json) as f:
            final = json.load(f)
        best_f1 = final.get("best_f1", float("nan"))
        final_metrics = final.get("final_round_metrics", {})
    
    # Get disparity for disability (primary focus)
    mean_disparity = final_metrics.get("mean_fairness_disparity_disability", float("nan"))
    if pd.isna(mean_disparity):
        # Fallback if key name changed
        mean_disparity = final_metrics.get("mean_fairness_disparity", float("nan"))
    
    results.append({
        "lambda": lam,
        "best_f1": best_f1,
        "d_fair_disability": mean_disparity,
    })

if not results:
    print("Error: No results collected.")
    exit(1)

df = pd.DataFrame(results).sort_values("lambda")
Path("experiments/results").mkdir(parents=True, exist_ok=True)
df.to_csv("experiments/results/lambda_tradeoff.csv", index=False)
print(df.to_string(index=False))

# Plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df["d_fair_disability"], df["best_f1"], 
                     c=df["lambda"], cmap="viridis", s=120, zorder=5, edgecolor='black', alpha=0.8)

# Connect points with a line to show Pareto frontier
ax.plot(df["d_fair_disability"], df["best_f1"], 'k--', alpha=0.3, zorder=1)

for _, row in df.iterrows():
    ax.annotate(f"λ={row['lambda']:.2f}", 
                (row["d_fair_disability"], row["best_f1"]),
                textcoords="offset points", xytext=(8, 8), fontsize=10)

cbar = plt.colorbar(scatter, label="Fairness Penalty (λ)")
ax.set_xlabel(r"Fairness Disparity ($D_{fair}$) — Lower is Better", fontsize=12)
ax.set_ylabel("Maximum F1 Score — Higher is Better", fontsize=12)
ax.set_title("Fairness-Accuracy Tradeoff (Pareto Frontier)", fontsize=14, fontweight="bold")
ax.grid(True, linestyle=':', alpha=0.6)

fig.savefig("experiments/figures/lambda_tradeoff.pdf", dpi=300, bbox_inches="tight")
print("Figure saved: experiments/figures/lambda_tradeoff.pdf")
