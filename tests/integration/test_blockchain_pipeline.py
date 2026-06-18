"""Integration tests for the full consent-to-audit blockchain pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.blockchain.chain import PermissionedBlockchain


@pytest.mark.integration
class TestBlockchainPipelineIntegration:

    def test_full_federated_round_audit_trail(self):
        """Simulate a complete 3-round, 4-node audit trail and verify it."""
        bc = PermissionedBlockchain(records_per_block=8)
        nodes = ["saudi_arabia", "united_states", "china", "europe"]
        n_rounds = 3

        for rnd in range(1, n_rounds + 1):
            for node_id in nodes:
                update = np.random.randn(50).astype(np.float32)
                bc.submit_model_update_hash(
                    protected_update=update,
                    node_id=node_id,
                    round_number=rnd,
                    consent_ref=f"consent_{node_id}_v1",
                    policy_ref=f"policy_{node_id}",
                )

        assert bc.get_record_count() == n_rounds * len(nodes)
        assert bc.verify_chain_integrity() is True
        assert bc.get_hash_completeness() == pytest.approx(1.0)

    def test_audit_export_roundtrip(self, tmp_path):
        """Export and reload audit log; data should be consistent."""
        bc = PermissionedBlockchain(records_per_block=3)
        for i in range(6):
            bc.submit_model_update_hash(
                np.ones(10, dtype=np.float32) * i, f"node_{i % 2}", i, f"cr{i}", "pr"
            )
        export_path = tmp_path / "audit.json"
        bc.export_audit_log(export_path)
        with open(export_path) as f:
            data = json.load(f)
        assert data["total_records"] == 6
        assert data["chain_integrity_valid"] is True

    def test_governance_events_in_audit_trail(self):
        """Governance events should be recorded alongside model update hashes."""
        bc = PermissionedBlockchain(records_per_block=5)
        bc.submit_model_update_hash(np.zeros(20, dtype=np.float32), "node_sa", 1, "cr1", "pr1")
        bc.submit_governance_event(
            node_id="node_sa",
            user_id_ref="anon_ref_007",
            risk_score=0.88,
            reviewer_action="flagged_for_review",
            round_number=1,
        )
        bc.submit_model_update_hash(np.zeros(20, dtype=np.float32), "node_us", 1, "cr2", "pr2")
        assert bc.verify_chain_integrity() is True
