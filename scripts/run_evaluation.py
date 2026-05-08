#!/usr/bin/env python3
"""Full evaluation pipeline — loads trained models and computes all paper metrics.

This script MUST be run AFTER run_federated_simulation.py and run_baselines.py.
It reads checkpoints from experiments/runs/, evaluates on held-out test data,
and writes verified CSV files to experiments/results/.

Usage:
    python scripts/run_evaluation.py \\
        --runs-dir experiments/runs/ \\
        --results-dir experiments/results/ \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EmploymentDataset
from src.evaluation.fairness_evaluator import FairnessEvaluator
from src.evaluation.metrics import aggregate_metrics_across_nodes, compute_full_metrics
from src.models.employment_model import EmploymentMatchingModel
from src.utils.config import load_config
from src.utils.io_utils import ensure_dir, save_json
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed_utils import set_global_seed

logger = get_logger("run_evaluation")

NODES = ["saudi_arabia", "united_states", "china", "europe"]


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run FedAgent-Chain evaluation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runs-dir",    type=str, default="experiments/runs/",
                   help="Directory containing simulation run outputs.")
    p.add_argument("--results-dir", type=str, default="experiments/results/",
                   help="Directory to save evaluation CSV results.")
    p.add_argument("--data-dir",    type=str, default="data/synthetic",
                   help="Root directory of the synthetic dataset.")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--seed-subdir", action="store_true",
                   help="Save results to seed-specific subdirectory for multi-seed aggregation.")
    p.add_argument("--log-level",   type=str, default="INFO")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_run_dir(runs_dir: Path, experiment_name: str, seed: int | None = None) -> Path | None:
    """Return the most recent run directory matching experiment_name and optionally seed.
    
    Filters out directories that do not contain a checkpoints/ folder with at least
    one .pt file, allowing fallback to older successful runs if the most recent one failed.
    """
    pattern = f"{experiment_name}_*"
    if seed is not None:
        pattern = f"{experiment_name}_seed{seed}_*"
    
    candidates = sorted(
        runs_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    
    for candidate in candidates:
        ckpt_dir = candidate / "checkpoints"
        if ckpt_dir.exists() and any(ckpt_dir.glob("*.pt")):
            return candidate
            
    return None


def _find_best_checkpoint(run_dir: Path) -> Path:
    """Return path to the best-performing checkpoint by F1, or final_model.pt."""
    per_round_path = run_dir / "metrics" / "per_round.json"
    final_path = run_dir / "checkpoints" / "final_model.pt"

    if not per_round_path.exists():
        return final_path

    with open(per_round_path, encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        return final_path

    # Find the round with the highest mean_f1
    best_round_data = max(history, key=lambda r: r.get("mean_f1", 0.0))
    best_round = best_round_data.get("round", len(history))

    # Checkpoints are saved every 5 rounds — find nearest saved round
    ckpt_dir = run_dir / "checkpoints"
    # Prefer best round checkpoint if it exists
    candidate = ckpt_dir / f"global_model_round_{best_round:03d}.pt"
    if candidate.exists():
        logger.info(
            "Using best-round checkpoint",
            round=best_round,
            f1=round(best_round_data.get("mean_f1", 0.0), 4),
        )
        return candidate

    # Fall back to the nearest saved checkpoint <= best_round
    saved = sorted(ckpt_dir.glob("global_model_round_*.pt"))
    saved_rounds = []
    for p in saved:
        try:
            saved_rounds.append((int(p.stem.split("_")[-1]), p))
        except ValueError:
            pass
    below = [(r, p) for r, p in saved_rounds if r <= best_round]
    if below:
        rnd, path = max(below, key=lambda x: x[0])
        logger.info("Using nearest saved checkpoint", round=rnd, requested_best=best_round)
        return path
    elif saved_rounds:
        rnd, path = min(saved_rounds, key=lambda x: x[0])
        logger.info("Using nearest available checkpoint (later than best)", round=rnd, requested_best=best_round)
        return path

    return final_path


def load_model_from_checkpoint(
    run_dir: Path,
    cfg_path: Path,
) -> EmploymentMatchingModel:
    """Load EmploymentMatchingModel from the best available checkpoint in run_dir."""
    cfg = load_config(cfg_path)
    model = EmploymentMatchingModel.from_config(cfg.get("model", {}))
    ckpt_path = _find_best_checkpoint(run_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run run_federated_simulation.py first."
        )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Wrap any 0-d scalars to avoid torch.from_numpy issues on re-load
    safe_state = {k: torch.as_tensor(v) for k, v in state_dict.items()}
    model.load_state_dict(safe_state, strict=False)
    model.eval()
    logger.info("Model loaded", checkpoint=str(ckpt_path))
    return model


def evaluate_model_on_node(
    model: EmploymentMatchingModel,
    node_id: str,
    seed: int,
    data_dir: Path = Path("data/synthetic"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EmploymentDataset]:
    """Return (y_true, y_pred, y_scores, test_dataset) for one node.

    Note: If eval-mode produces NaN probs (a known issue when BatchNorm running
    statistics are corrupted by FedAvg delta accumulation), automatically falls
    back to batch-statistics mode (model.train() + no_grad) which recomputes
    running stats per batch, avoiding the NaN issue.
    """
    node_dir = data_dir / node_id
    users_df    = pd.read_csv(node_dir / "users.csv")
    jobs_df     = pd.read_csv(node_dir / "jobs.csv")
    outcomes_df = pd.read_csv(node_dir / "outcomes.csv")

    full_ds = EmploymentDataset(
        outcomes_df=outcomes_df,
        users_df=users_df,
        jobs_df=jobs_df,
        consent_filter=True,
    )
    _, test_ds = full_ds.split(test_size=0.20, seed=seed)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    def _run_inference(train_mode: bool) -> tuple[list, list, list]:
        if train_mode:
            model.train()   # Recompute BatchNorm stats per batch
        else:
            model.eval()    # Use running stats (may be corrupted by FedAvg)
        y_t, y_p, y_s = [], [], []
        with torch.no_grad():
            for batch in loader:
                feats  = batch["features"]
                labels = batch["label"].numpy()
                probs  = model(feats).squeeze(-1).numpy()
                preds  = (probs >= 0.5).astype(int)
                y_t.extend(labels.tolist())
                y_p.extend(preds.tolist())
                y_s.extend(probs.tolist())
        return y_t, y_p, y_s

    y_true_list, y_pred_list, y_score_list = _run_inference(train_mode=False)

    # Detect BatchNorm corruption: running stats diverged from FedAvg accumulation
    probs_arr = np.array(y_score_list)
    if np.isnan(probs_arr).any() or np.isinf(probs_arr).any():
        logger.warning(
            "NaN/Inf probs detected — BatchNorm running stats corrupted by FedAvg. "
            "Falling back to batch-statistics mode.",
            node=node_id,
        )
        y_true_list, y_pred_list, y_score_list = _run_inference(train_mode=True)
        probs_arr = np.array(y_score_list)

    return (
        np.array(y_true_list),
        np.array(y_pred_list),
        probs_arr,
        test_ds,
    )


def build_prediction_dataframe(
    model: EmploymentMatchingModel,
    node_id: str,
    seed: int,
    data_dir: Path = Path("data/synthetic"),
) -> pd.DataFrame:
    """Return a DataFrame with predictions and user attributes for fairness eval."""
    y_true, y_pred, y_scores, test_ds = evaluate_model_on_node(
        model, node_id, seed, data_dir
    )
    outcomes = test_ds.outcomes.copy()
    outcomes["predicted_label"] = y_pred
    outcomes["predicted_score"] = y_scores

    # Merge user attributes needed by FairnessEvaluator
    users_df = test_ds.users_df.reset_index()
    merged = outcomes.merge(
        users_df[["user_id", "disability_category",
                  "language_primary", "preferred_work_mode"]],
        on="user_id",
        how="left",
    )
    merged["node_id"] = node_id
    return merged


# ── Table generators ──────────────────────────────────────────────────────────

def generate_table_2(
    models: dict[str, EmploymentMatchingModel],
    seed: int,
    results_dir: Path,
    data_dir: Path,
) -> pd.DataFrame:
    """Table 2: Classification and ranking metrics for all methods."""
    rows = []
    for method_name, model in models.items():
        node_metrics: dict[str, dict] = {}
        for i, node_id in enumerate(NODES):
            y_true, y_pred, y_scores, _ = evaluate_model_on_node(
                model, node_id, seed + i * 1000, data_dir
            )
            node_metrics[node_id] = compute_full_metrics(
                y_true, y_pred, y_scores, k_values=[5, 10]
            )
            logger.info(
                "Node evaluated",
                method=method_name,
                node=node_id,
                f1=round(node_metrics[node_id]["f1"], 4),
                accuracy=round(node_metrics[node_id]["accuracy"], 4),
            )

        agg = aggregate_metrics_across_nodes(node_metrics)
        rows.append({
            "Method":    method_name,
            "Accuracy":  round(agg["mean_accuracy"],  4),
            "Precision": round(agg["mean_precision"], 4),
            "Recall":    round(agg["mean_recall"],    4),
            "F1":        round(agg["mean_f1"],        4),
            "F1_std":    round(agg["std_f1"],         4),
            "P@5":       round(agg.get("mean_precision_at_5", 0.0), 4),
            "R@5":       round(agg.get("mean_recall_at_5",    0.0), 4),
            "P@10":      round(agg.get("mean_precision_at_10", 0.0), 4),
            "R@10":      round(agg.get("mean_recall_at_10",   0.0), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "table_2_model_performance.csv", index=False)
    logger.info("Table 2 saved (computed from checkpoints)")
    return df


def generate_table_3(
    models: dict[str, EmploymentMatchingModel],
    seed: int,
    results_dir: Path,
    data_dir: Path,
) -> pd.DataFrame:
    """Table 3: Fairness disparity across protected attributes."""
    evaluator = FairnessEvaluator()
    rows = []

    attr_display = {
        "disability_category": "Disability Category",
        "language_primary":    "Language Group",
        "preferred_work_mode": "Work Mode",
        "node_id":             "Regional Node",
    }

    for attr_key, attr_label in attr_display.items():
        row: dict = {"Attribute": attr_label}
        for method_name, model in models.items():
            all_preds = []
            for i, node_id in enumerate(NODES):
                df_node = build_prediction_dataframe(
                    model, node_id, seed + i * 1000, data_dir
                )
                all_preds.append(df_node)
            combined = pd.concat(all_preds, ignore_index=True)
            disparities = evaluator.evaluate(
                combined,
                y_true_col="suitability_label",
                y_pred_col="predicted_label",
            )
            row[method_name] = round(disparities.get(attr_key, float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Compute relative reduction where both Standard FedAvg and FedAgent-Chain exist
    if "Standard FedAvg" in df.columns and "FedAgent-Chain" in df.columns:
        df["Reduction"] = df.apply(
            lambda r: (
                f"{100*(r['Standard FedAvg'] - r['FedAgent-Chain']) / (r['Standard FedAvg'] + 1e-9):.1f}%"
                if pd.notna(r["Standard FedAvg"]) and pd.notna(r["FedAgent-Chain"])
                else "N/A"
            ),
            axis=1,
        )
    df.to_csv(results_dir / "table_3_fairness_results.csv", index=False)
    logger.info("Table 3 saved (computed from predictions)")
    return df


def generate_table_4_blockchain(run_dir: Path, results_dir: Path) -> pd.DataFrame:
    """Table 4: Read blockchain metrics from the simulation audit log."""
    audit_path = run_dir / "blockchain_logs" / "audit_trail.json"
    if not audit_path.exists():
        logger.warning("Audit trail not found", path=str(audit_path))
        return pd.DataFrame()

    with open(audit_path, encoding="utf-8") as f:
        audit = json.load(f)

    records = []
    for block in audit.get("blocks", []):
        for r in block.get("records", []):
            if isinstance(r, dict) and r.get("type") != "genesis":
                records.append(r)

    total        = len(records)
    valid_hashes = sum(
        1 for r in records
        if isinstance(r.get("hash", ""), str) and len(r["hash"]) == 64
    )
    completeness = valid_hashes / total if total > 0 else 0.0

    rows = [
        {
            "Metric": "Hash Completeness",
            "Value": f"{completeness*100:.1f}%",
            "Description": "Fraction of records with valid SHA-256 hash",
        },
        {
            "Metric": "Chain Integrity",
            "Value": "Valid" if audit.get("chain_integrity_valid") else "INVALID",
            "Description": "SHA-256 hash chain verification result",
        },
        {
            "Metric": "Total Audit Records",
            "Value": str(total),
            "Description": "Model update hashes submitted",
        },
        {
            "Metric": "Chain Length (blocks)",
            "Value": str(audit.get("chain_length", 0)),
            "Description": "Number of finalized blocks",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "table_4_blockchain_results.csv", index=False)
    logger.info("Table 4 saved (from audit_trail.json)")
    return df


def generate_table_7_overhead(run_dir: Path, results_dir: Path) -> pd.DataFrame:
    """Table 7: Read per-round timing from per_round.json."""
    metrics_path = run_dir / "metrics" / "per_round.json"
    if not metrics_path.exists():
        logger.warning("per_round.json not found", path=str(metrics_path))
        return pd.DataFrame()

    with open(metrics_path, encoding="utf-8") as f:
        rounds = json.load(f)

    if not rounds:
        return pd.DataFrame()

    durations = [r.get("duration_seconds", 0.0) for r in rounds]
    rows = [
        {
            "Component": "Avg round duration (4 nodes)",
            "CPU Time": f"{np.mean(durations):.1f}s",
            "Notes": "Mean over all federated rounds",
        },
        {
            "Component": "Min round duration",
            "CPU Time": f"{np.min(durations):.1f}s",
            "Notes": "",
        },
        {
            "Component": "Max round duration",
            "CPU Time": f"{np.max(durations):.1f}s",
            "Notes": "",
        },
        {
            "Component": "Total simulation time",
            "CPU Time": f"{sum(durations):.1f}s",
            "Notes": f"({len(rounds)} rounds x 4 nodes)",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "table_7_overhead.csv", index=False)
    logger.info("Table 7 saved (from per_round.json)")
    return df


def generate_table_5_agents(
    seed: int,
    results_dir: Path,
    n_eval_users: int = 200,
) -> pd.DataFrame:
    """Table 5: Agentic AI service evaluation on synthetic test users."""
    from omegaconf import OmegaConf
    from src.agents.employment_agent import EmploymentAgent
    from src.agents.governance_agent import GovernanceAgent
    from src.agents.upskilling_agent import UpskillingAgent
    from src.agents.accommodation_agent import AccommodationAgent
    from src.agents.multilingual_agent import MultilingualCommunicationAgent
    from src.data.synthetic_generator import (
        generate_user_profiles, generate_job_profiles
    )
    from src.utils.seed_utils import get_rng

    rng = get_rng(seed)

    # Load config defaults
    cfg = OmegaConf.create({
        "alpha": 0.40, "beta": 0.25, "gamma": 0.20, "delta": 0.15, "top_k": 10,
        "top_k_skills": 5, "review_threshold": 0.38,
    })
    gov_cfg = OmegaConf.create({"review_threshold": 0.38})

    emp_agent   = EmploymentAgent(cfg, governance_threshold=0.70)
    gov_agent   = GovernanceAgent(gov_cfg)
    ups_agent   = UpskillingAgent(cfg, governance_threshold=0.70)
    acc_agent   = AccommodationAgent(cfg, governance_threshold=0.70)
    lang_agent  = MultilingualCommunicationAgent(cfg, governance_threshold=0.70)

    users = generate_user_profiles("united_states", n_eval_users, rng)
    users = [u for u in users if u.consent_given]
    jobs  = generate_job_profiles("united_states", 100, rng)

    emp_scores, gov_detections, gov_fps = [], [], []
    ups_coverages, acc_coverages, lang_adequacies = [], [], []

    for user in users[:n_eval_users]:
        emp_out   = emp_agent.run(user_id=user.user_id, user=user, jobs=jobs)
        gov_out   = gov_agent.run(
            user_id=user.user_id,
            employment_output=emp_out,
            disability_category=user.disability_category.value,
        )
        ups_out   = ups_agent.run(
            user_id=user.user_id, user=user,
            top_jobs=jobs[: min(10, len(jobs))]
        )

        top_job_profile = None
        if emp_out.recommendations:
            top_job_id = emp_out.recommendations[0].get("job_id")
            matched    = [j for j in jobs if j.job_id == top_job_id]
            if matched:
                top_job_profile = matched[0]

        if top_job_profile:
            acc_out = acc_agent.run(
                user_id=user.user_id, user=user, job=top_job_profile
            )
            lang_out = lang_agent.run(
                user_id=user.user_id, user=user,
                job_language=top_job_profile.language_required,
            )
            acc_coverages.append(acc_out.metadata.get("coverage", 0.0))
            lang_adequacies.append(lang_out.confidence)

        emp_scores.append(emp_out.confidence)
        # Governance: high-risk if risk_score > threshold
        is_high_risk  = gov_out.risk_score > 0.38
        # We treat low-confidence + multiple disability as ground-truth high-risk
        gt_high_risk  = (
            emp_out.confidence < 0.55
            or user.disability_category.value == "multiple"
        )
        if gt_high_risk:
            gov_detections.append(1 if is_high_risk else 0)
        else:
            gov_fps.append(1 if is_high_risk else 0)

        if ups_out.recommendations:
            ups_coverages.append(
                ups_out.confidence
            )

    rows = [
        {"Agent": "Employment Matching", "Metric": "Mean Confidence",
         "Score": round(float(np.mean(emp_scores)), 4)},
        {"Agent": "Upskilling", "Metric": "Skill Gap Coverage",
         "Score": round(float(np.mean(ups_coverages)) if ups_coverages else 0.0, 4)},
        {"Agent": "Accommodation", "Metric": "Accommodation Coverage",
         "Score": round(float(np.mean(acc_coverages)) if acc_coverages else 0.0, 4)},
        {"Agent": "Multilingual", "Metric": "Language Adequacy",
         "Score": round(float(np.mean(lang_adequacies)) if lang_adequacies else 0.0, 4)},
        {"Agent": "Governance", "Metric": "High-Risk Detection Rate",
         "Score": round(float(np.mean(gov_detections)) if gov_detections else 0.0, 4)},
        {"Agent": "Governance", "Metric": "False Positive Rate",
         "Score": round(float(np.mean(gov_fps)) if gov_fps else 0.0, 4)},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "table_5_agent_results.csv", index=False)
    logger.info("Table 5 saved (computed from agent runs)")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level, format="console")
    set_global_seed(args.seed)

    runs_dir    = Path(args.runs_dir)
    data_dir    = Path(args.data_dir)

    if args.seed_subdir:
        results_dir = ensure_dir(Path(args.results_dir) / f"seed_{args.seed}")
    else:
        results_dir = ensure_dir(Path(args.results_dir))

    logger.info(
        "Starting evaluation pipeline",
        runs_dir=str(runs_dir),
        results_dir=str(results_dir),
    )

    # ── Locate simulation run directories ─────────────────────────────────────
    run_map = {
        "FedAgent-Chain":  find_run_dir(runs_dir, "fedagent_chain_full", args.seed),
        "Standard FedAvg": find_run_dir(runs_dir, "ablation_no_fairness", args.seed),
        "Local Baseline":  find_run_dir(runs_dir, "baseline_local", args.seed),
        "Centralized":     find_run_dir(runs_dir, "baseline_centralized", args.seed),
    }

    missing = [k for k, v in run_map.items() if v is None]
    if missing:
        logger.error(
            "Run directories not found — execute simulations first",
            missing=missing,
        )
        raise FileNotFoundError(
            f"Missing run directories for: {missing}\n"
            "Run: python scripts/run_federated_simulation.py (and run_baselines.py)"
        )

    for method, run_dir in run_map.items():
        logger.info("Found run directory", method=method, path=str(run_dir))

    # ── Load configs and model checkpoints ─────────────────────────────────────
    cfg_map = {
        "FedAgent-Chain":  Path("configs/experiment/fedagent_chain_full.yaml"),
        "Standard FedAvg": Path("configs/experiment/ablation/no_fairness.yaml"),
        "Local Baseline":  Path("configs/experiment/baseline_local.yaml"),
        "Centralized":     Path("configs/experiment/baseline_centralized.yaml"),
    }

    models: dict[str, EmploymentMatchingModel] = {}
    for method_name, run_dir in run_map.items():
        logger.info("Loading model", method=method_name, run_dir=str(run_dir))
        models[method_name] = load_model_from_checkpoint(
            run_dir, cfg_map[method_name]
        )

    # ── Generate tables ────────────────────────────────────────────────────────
    logger.info("Generating Table 2 (classification metrics)...")
    t2 = generate_table_2(models, args.seed, results_dir, data_dir)

    logger.info("Generating Table 3 (fairness disparity)...")
    t3 = generate_table_3(models, args.seed, results_dir, data_dir)

    logger.info("Generating Table 4 (blockchain audit)...")
    t4 = generate_table_4_blockchain(run_map["FedAgent-Chain"], results_dir)

    logger.info("Generating Table 7 (overhead)...")
    t7 = generate_table_7_overhead(run_map["FedAgent-Chain"], results_dir)

    logger.info("Generating Table 5 (agentic AI services)...")
    t5 = generate_table_5_agents(args.seed, results_dir)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(" [OK] Evaluation complete -- all tables computed from checkpoints")
    print(f"{'='*65}")
    print("\nTable 2 -- Model Performance:")
    print(t2.to_string(index=False))
    if not t3.empty:
        print("\nTable 3 -- Fairness Disparity:")
        print(t3.to_string(index=False))
    if not t4.empty:
        print("\nTable 4 -- Blockchain Audit:")
        print(t4.to_string(index=False))

    print("\nTable 5 — Agentic AI Services:")
    print(t5.to_string(index=False))

    print(f"\nResults saved to: {results_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
