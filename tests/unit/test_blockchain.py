"""Unit tests for the permissioned blockchain simulation."""

from __future__ import annotations

import numpy as np
import pytest


class TestPermissionedBlockchain:
    """Tests for the PermissionedBlockchain class."""

    def test_genesis_block_created(self, blockchain):
        """Blockchain should have one block (genesis) on initialization."""
        assert len(blockchain._chain) == 1
        assert blockchain._chain[0].block_index == 0

    def test_submit_model_update_hash_returns_record(self, blockchain):
        """submit_model_update_hash should return a valid BlockchainRecord."""
        update = np.zeros(100, dtype=np.float32)
        record = blockchain.submit_model_update_hash(
            protected_update=update,
            node_id="saudi_arabia",
            round_number=1,
            consent_ref="consent_ref_001",
            policy_ref="policy_gdpr",
        )
        assert record.node_id == "saudi_arabia"
        assert record.round_number == 1
        assert len(record.hash) == 64
        assert record.status == "accepted"

    def test_hash_is_deterministic(self, blockchain):
        """Same inputs must produce the same hash."""
        update = np.ones(50, dtype=np.float32)
        h1 = blockchain._compute_update_hash(update, "node_sa", 1)
        h2 = blockchain._compute_update_hash(update, "node_sa", 1)
        assert h1 == h2

    def test_hash_sensitive_to_node_id(self, blockchain):
        """Different node IDs must produce different hashes."""
        update = np.ones(50, dtype=np.float32)
        h1 = blockchain._compute_update_hash(update, "node_sa", 1)
        h2 = blockchain._compute_update_hash(update, "node_us", 1)
        assert h1 != h2

    def test_hash_sensitive_to_round_number(self, blockchain):
        """Different round numbers must produce different hashes."""
        update = np.ones(50, dtype=np.float32)
        h1 = blockchain._compute_update_hash(update, "node_sa", 1)
        h2 = blockchain._compute_update_hash(update, "node_sa", 2)
        assert h1 != h2

    def test_hash_is_valid_sha256_format(self, blockchain):
        """Hash must be a 64-character lowercase hexadecimal string."""
        update = np.zeros(10, dtype=np.float32)
        h = blockchain._compute_update_hash(update, "test_node", 0)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_no_raw_data_stored_in_record(self, blockchain):
        """Blockchain records must NOT contain raw disability data fields."""
        update = np.random.randn(100).astype(np.float32)
        record = blockchain.submit_model_update_hash(
            protected_update=update,
            node_id="node_sa",
            round_number=1,
            consent_ref="consent_ref_001",
            policy_ref="policy_gdpr_art9",
        )
        record_dict = record.model_dump()
        forbidden_fields = [
            "disability_type",
            "user_id",
            "skill_vector",
            "accommodation_needs",
            "raw_data",
            "personal_data",
        ]
        for field in forbidden_fields:
            assert (
                field not in record_dict
            ), f"Forbidden field '{field}' found in blockchain record!"

    def test_chain_integrity_valid_after_submissions(self, blockchain):
        """Chain integrity should remain valid after multiple submissions."""
        for i in range(12):  # Trigger block finalization
            blockchain.submit_model_update_hash(
                protected_update=np.ones(10, dtype=np.float32) * i,
                node_id=f"node_{i % 4}",
                round_number=i,
                consent_ref=f"consent_{i}",
                policy_ref="policy_gdpr",
            )
        assert blockchain.verify_chain_integrity() is True

    def test_record_count_increases(self, blockchain):
        """get_record_count() should increase with each submission."""
        assert blockchain.get_record_count() == 0
        blockchain.submit_model_update_hash(
            np.zeros(10, dtype=np.float32), "node_sa", 1, "cr1", "pr1"
        )
        assert blockchain.get_record_count() == 1
        blockchain.submit_model_update_hash(
            np.zeros(10, dtype=np.float32), "node_us", 1, "cr2", "pr2"
        )
        assert blockchain.get_record_count() == 2

    def test_hash_completeness_all_valid(self, blockchain):
        """Hash completeness should be 1.0 when all records have valid hashes."""
        for i in range(5):
            blockchain.submit_model_update_hash(
                np.ones(10, dtype=np.float32), f"node_{i}", i, f"cr{i}", "pr"
            )
        assert blockchain.get_hash_completeness() == pytest.approx(1.0)

    def test_export_audit_log_creates_file(self, blockchain, tmp_path):
        """export_audit_log should create a valid JSON file."""
        import json

        blockchain.submit_model_update_hash(
            np.zeros(10, dtype=np.float32), "node_sa", 1, "cr1", "pr1"
        )
        output_path = tmp_path / "audit.json"
        blockchain.export_audit_log(output_path)
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert "chain_length" in data
        assert "total_records" in data
        assert data["total_records"] == 1

    def test_governance_event_recorded(self, blockchain):
        """Governance events should be stored in pending records."""
        record_id = blockchain.submit_governance_event(
            node_id="node_sa",
            user_id_ref="anon_ref_001",
            risk_score=0.85,
            reviewer_action="approve_with_modifications",
            round_number=3,
        )
        assert isinstance(record_id, str)
        assert len(record_id) > 0
