
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add repo root to path
sys.path.insert(0, str(Path.cwd()))

from src.models.employment_model import EmploymentMatchingModel
from src.data.dataset import EmploymentDataset

def debug_evaluation():
    # Load model
    model = EmploymentMatchingModel()
    ckpt_path = Path("experiments/runs/ablation_no_fairness_seed123_20260508_164950/checkpoints/global_model_round_001.pt")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load data for Saudi Arabia
    node_id = "saudi_arabia"
    data_dir = Path("data/synthetic")
    users_df = pd.read_csv(data_dir / node_id / "users.csv")
    jobs_df = pd.read_csv(data_dir / node_id / "jobs.csv")
    outcomes_df = pd.read_csv(data_dir / node_id / "outcomes.csv")
    
    full_ds = EmploymentDataset(outcomes_df, users_df, jobs_df, consent_filter=True)
    # Split with seed 123 (matching node 0)
    _, test_ds = full_ds.split(test_size=0.2, seed=123)
    
    # Run inference
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=256)
    
    y_true, y_pred = [], []
    for batch in loader:
        features = batch["features"]
        labels = batch["label"].numpy()
        with torch.no_grad():
            preds = model.predict(features).numpy()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    print(f"Debug evaluation for Saudi Arabia (Seed 123):")
    print(f"F1: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Prediction distribution: {np.bincount(y_pred.astype(int))}")
    print(f"True distribution: {np.bincount(y_true.astype(int))}")

if __name__ == "__main__":
    debug_evaluation()
