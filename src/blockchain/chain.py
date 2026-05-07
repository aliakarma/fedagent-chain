"""Permissioned blockchain simulation for FedAgent-Chain audit layer.

Implements a lightweight in-memory permissioned blockchain for recording:
- Model update hashes (cryptographic fingerprints, NOT raw updates)
- Consent validation records
- Governance review events
- Policy compliance records

The blockchain simulation can be replaced with Hyperledger Fabric by
substituting this module while keeping the public API identical.

IMPORTANT: This blockchain never stores raw disability-employment data.
It stores only hashes, references, and metadata.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.data.schema import BlockchainRecord
from src.utils.logging_utils import get_logger

logger = get_logger("PermissionedBlockchain")


@dataclass
class Block:
    """A single block in the permissioned blockchain."""

    block_index: int
    previous_hash: str
    timestamp: str
    records: List[Dict]
    nonce: int = 0
    block_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.block_hash = self._compute_block_hash()

    def _compute_block_hash(self) -> str:
        """Compute the SHA-256 hash of this block's content."""
        block_content = json.dumps(
            {
                "block_index": self.block_index,
                "previous_hash": self.previous_hash,
                "timestamp": self.timestamp,
                "records": self.records,
                "nonce": self.nonce,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(block_content).hexdigest()

    def verify(self) -> bool:
        """Verify the block's hash matches its content."""
        return self.block_hash == self._compute_block_hash()


class PermissionedBlockchain:
    """Lightweight permissioned blockchain for FedAgent-Chain audit logging.

    Stores model update hashes, consent metadata, and governance records
    in a tamper-evident append-only chain.

    Hash construction per paper Section 4.8:
        h_k^t = H(Δw̃_k || ID_k || t)

    where H is SHA-256, Δw̃_k is the privacy-protected update,
    ID_k is the node identifier, and t is the round number.

    Parameters
    ----------
    records_per_block : int
        Number of audit records to include per block.
    storage_path : str or Path, optional
        If provided, the chain is persisted to disk at this path.
    """

    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        records_per_block: int = 10,
        storage_path: Optional[str | Path] = None,
    ) -> None:
        self.records_per_block = records_per_block
        self.storage_path = Path(storage_path) if storage_path else None
        self._chain: List[Block] = []
        self._pending_records: List[Dict] = []
        self._all_records: List[BlockchainRecord] = []

        # Create genesis block
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """Create and append the genesis (first) block."""
        genesis = Block(
            block_index=0,
            previous_hash=self.GENESIS_HASH,
            timestamp=datetime.now(timezone.utc).isoformat(),
            records=[{"type": "genesis", "message": "FedAgent-Chain blockchain initialized"}],
        )
        self._chain.append(genesis)
        logger.info("Genesis block created", block_hash=genesis.block_hash[:16] + "...")

    def _compute_update_hash(
        self,
        protected_update: np.ndarray,
        node_id: str,
        round_number: int,
    ) -> str:
        """Compute h_k^t = H(Δw̃_k || ID_k || t) as specified in Section 4.8.

        Parameters
        ----------
        protected_update : np.ndarray
            Privacy-protected model update (NOT raw gradients).
        node_id : str
            Identifier of the submitting institutional node.
        round_number : int
            Current federated learning round number.

        Returns
        -------
        str
            64-character hexadecimal SHA-256 hash.
        """
        # Hash the raw bytes of the protected update
        update_hash = hashlib.sha256(protected_update.tobytes()).hexdigest()
        # Combine with node ID and round number
        combined = f"{update_hash}{node_id}{round_number}".encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def submit_model_update_hash(
        self,
        protected_update: np.ndarray,
        node_id: str,
        round_number: int,
        consent_ref: str,
        policy_ref: str,
    ) -> BlockchainRecord:
        """Submit a model update hash to the audit chain.

        This is the primary method called after each local training round.
        Only the cryptographic hash of the protected update is recorded.
        Raw disability data and model weights are never stored.

        Parameters
        ----------
        protected_update : np.ndarray
            Privacy-protected model update tensor.
        node_id : str
            Identifier of the submitting node.
        round_number : int
            Federated learning round number.
        consent_ref : str
            Reference ID to the consent record for this node's data.
        policy_ref : str
            Applicable data governance policy (e.g., "policy_gdpr_art9").

        Returns
        -------
        BlockchainRecord
            The created and validated audit record.
        """
        h = self._compute_update_hash(protected_update, node_id, round_number)

        record = BlockchainRecord(
            record_id=str(uuid.uuid4()),
            node_id=node_id,
            hash=h,
            consent_ref=consent_ref,
            policy_ref=policy_ref,
            round_number=round_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="accepted",
        )

        self._all_records.append(record)
        self._pending_records.append(record.model_dump())

        if len(self._pending_records) >= self.records_per_block:
            self._finalize_block()

        logger.info(
            "Model update hash submitted",
            node_id=node_id,
            round_number=round_number,
            hash_prefix=h[:16] + "...",
            consent_ref=consent_ref,
        )
        return record

    def submit_governance_event(
        self,
        node_id: str,
        user_id_ref: str,
        risk_score: float,
        reviewer_action: str,
        round_number: int,
    ) -> str:
        """Record a human-in-the-loop governance review event.

        Parameters
        ----------
        node_id : str
            Node where the governance event occurred.
        user_id_ref : str
            Anonymized reference to the user involved (NOT the actual user_id).
        risk_score : float
            The computed risk score that triggered the review.
        reviewer_action : str
            Action taken by the human reviewer.
        round_number : int
            Current round number.

        Returns
        -------
        str
            Record ID of the governance event.
        """
        record = {
            "record_id": str(uuid.uuid4()),
            "type": "governance_event",
            "node_id": node_id,
            "user_id_ref": user_id_ref,
            "risk_score": round(risk_score, 4),
            "reviewer_action": reviewer_action,
            "round_number": round_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._pending_records.append(record)
        if len(self._pending_records) >= self.records_per_block:
            self._finalize_block()

        logger.info(
            "Governance event recorded",
            node_id=node_id,
            risk_score=risk_score,
            action=reviewer_action,
        )
        return record["record_id"]

    def _finalize_block(self) -> Block:
        """Create a new block with all pending records."""
        prev_hash = self._chain[-1].block_hash
        block = Block(
            block_index=len(self._chain),
            previous_hash=prev_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            records=self._pending_records.copy(),
        )
        self._chain.append(block)
        self._pending_records.clear()

        logger.info(
            "Block finalized",
            block_index=block.block_index,
            n_records=len(block.records),
            block_hash=block.block_hash[:16] + "...",
        )

        if self.storage_path:
            self._persist()

        return block

    def verify_chain_integrity(self) -> bool:
        """Verify that the entire blockchain has not been tampered with.

        Checks that each block's hash matches its content and that
        the chain of previous_hash references is unbroken.

        Returns
        -------
        bool
            True if the chain is intact, False if any block is corrupted.
        """
        for i, block in enumerate(self._chain):
            if not block.verify():
                logger.error(
                    "Block hash mismatch — possible tampering detected",
                    block_index=i,
                )
                return False

            if i > 0 and block.previous_hash != self._chain[i - 1].block_hash:
                logger.error(
                    "Chain link broken — possible insertion/deletion",
                    block_index=i,
                )
                return False

        logger.info("Chain integrity verified", n_blocks=len(self._chain))
        return True

    def get_record_count(self) -> int:
        """Return total number of submitted audit records."""
        return len(self._all_records)

    def get_all_records(self) -> List[BlockchainRecord]:
        """Return all model update hash records."""
        return self._all_records.copy()

    def export_audit_log(self, path: str | Path) -> None:
        """Export the complete blockchain audit log to JSON.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "chain_length": len(self._chain),
            "total_records": self.get_record_count(),
            "chain_integrity_valid": self.verify_chain_integrity(),
            "blocks": [
                {
                    "block_index": b.block_index,
                    "block_hash": b.block_hash,
                    "previous_hash": b.previous_hash,
                    "timestamp": b.timestamp,
                    "n_records": len(b.records),
                    "records": b.records,
                }
                for b in self._chain
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info("Audit log exported", path=str(path))

    def _persist(self) -> None:
        """Persist the current chain state to disk."""
        if self.storage_path is None:
            return
        self.export_audit_log(self.storage_path / "audit_log.json")

    def get_hash_completeness(self) -> float:
        """Compute hash completeness: fraction of records with valid hashes.

        Returns
        -------
        float
            Hash completeness in [0, 1]. Should be 1.0 in correct operation.
        """
        if not self._all_records:
            return 0.0
        valid = sum(
            1 for r in self._all_records
            if len(r.hash) == 64 and r.status == "accepted"
        )
        return valid / len(self._all_records)
