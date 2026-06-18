import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.blockchain.chain import PermissionedBlockchain
from src.data.dataset import EmploymentDataset
from src.data.synthetic_generator import generate_synthetic_node_data
from src.federated.client import FederatedClient


def test_batch_group_labels_alignment():
    """Verify that group labels returned by _get_batch_group_labels
    correspond to the correct samples via the idx field.
    Resolves CI-5 (group label mismatch).
    """
    # 1. Setup synthetic data and dataset
    data = generate_synthetic_node_data("saudi_arabia", 100, 50, 200, seed=42)
    print(f"User group labels head: \n{data['users']['disability_category'].head()}")
    ds = EmploymentDataset(data["outcomes"], data["users"], data["jobs"])

    # 2. Verify idx is returned by __getitem__
    item = ds[0]
    assert "idx" in item, "EmploymentDataset.__getitem__ must return 'idx'"
    assert item["idx"].item() == 0

    item = ds[42]
    assert item["idx"].item() == 42

    # 3. Verify idx survives DataLoader batching and shuffling
    # Use a generator to ensure shuffling is measurable
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(ds, batch_size=16, shuffle=True, generator=g)
    batch = next(iter(loader))
    assert "idx" in batch, "DataLoader must propagate 'idx' field"
    assert len(batch["idx"]) == 16

    batch_indices = batch["idx"].numpy().astype(int)
    print(f"Batch indices: {batch_indices}")

    # 4. Setup Client to test _get_batch_group_labels
    cfg = OmegaConf.create(
        {
            "federated": {"local_epochs": 1, "batch_size": 16, "learning_rate": 0.01},
            "privacy": {"enabled": False},
            "model": {"hidden_dims": [32]},
        }
    )
    blockchain = PermissionedBlockchain()
    client = FederatedClient(
        node_id="test_node", train_dataset=ds, test_dataset=ds, cfg=cfg, blockchain=blockchain
    )

    # 5. Verify group labels correspond to correct samples in the shuffled batch
    group_cache = ds.get_group_labels("disability_category")
    actual_groups = client._get_batch_group_labels(batch)

    print(f"Type of group labels: {type(actual_groups[0])}")
    print(f"Sample group label:   {actual_groups[0]}")
    print(f"Actual groups for batch: {actual_groups}")
    print(f"Group cache (first 16):  {group_cache[:16]}")

    expected_groups = group_cache[batch_indices]

    assert list(actual_groups) == list(expected_groups), (
        "Group labels do not match expected values for batch indices. "
        "This indicates a mismatch between shuffled batch samples and their group labels."
    )

    # 6. Verify that the OLD BUG would have failed
    # The old approach returned group_cache[:batch_size]
    old_approach_groups = group_cache[:16]
    # With shuffling, it's extremely likely the first 16 labels don't match the shuffled 16 indices
    # We check if they are identical (which they shouldn't be with high probability)
    match_count = sum(a == b for a, b in zip(old_approach_groups, actual_groups, strict=False))
    print(
        f"\nAlignment check: New approach matches {len(actual_groups)}/16. Old approach matched {match_count}/16."
    )

    if match_count < 16:
        print("SUCCESS: Confirmed that the old approach was misaligned.")
    else:
        # If it happens to match by pure luck, we don't fail the test but note it
        print("NOTE: Old approach happened to match by luck (1 in 10^lots probability).")


if __name__ == "__main__":
    test_batch_group_labels_alignment()
