"""Adaptive upskilling recommendation agent for FedAgent-Chain."""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig

from src.agents.base_agent import AgentOutput, BaseAgent
from src.data.schema import JobProfile, UserProfile


class UpskillingAgent(BaseAgent):
    """Recommends targeted skill development based on job-skill gap analysis.

    Computes the skill gap G(u_i, j_r) = required_skills AND NOT user_skills
    and prioritises the top-K missing skills that appear most frequently
    across the top-ranked job opportunities.

    Parameters
    ----------
    agent_cfg : DictConfig
        Config with `top_k_skills` (int, default 5).
    """

    def __init__(self, agent_cfg: DictConfig, governance_threshold: float = 0.70) -> None:
        super().__init__(agent_cfg, governance_threshold)
        self.top_k_skills: int = int(agent_cfg.get("top_k_skills", 5))

    def compute_skill_gap(self, user_skills: np.ndarray, job_skills: np.ndarray) -> np.ndarray:
        """Return binary vector of skills required by job but missing in user."""
        return np.maximum(0, job_skills - user_skills).astype(int)

    def aggregate_skill_gaps(
        self, user: UserProfile, jobs: list[JobProfile]
    ) -> list[dict[str, Any]]:
        """Aggregate skill gaps across top jobs and rank by frequency."""
        u_skills = np.array(user.skill_vector, dtype=float)
        gap_counts = np.zeros(50, dtype=int)

        for job in jobs:
            j_skills = np.array(job.required_skills, dtype=float)
            gap = self.compute_skill_gap(u_skills, j_skills)
            gap_counts += gap

        # Top-K most-needed skills
        top_indices = np.argsort(gap_counts)[::-1][: self.top_k_skills]
        return [
            {
                "skill_index": int(idx),
                "frequency": int(gap_counts[idx]),
                "priority": i + 1,
            }
            for i, idx in enumerate(top_indices)
            if gap_counts[idx] > 0
        ]

    def run(
        self,
        user_id: str,
        user: UserProfile | None = None,
        top_jobs: list[JobProfile] | None = None,
        **kwargs: Any,
    ) -> AgentOutput:
        if user is None or top_jobs is None:
            raise ValueError("UpskillingAgent requires 'user' and 'top_jobs'.")

        skill_gaps = self.aggregate_skill_gaps(user, top_jobs)
        coverage = len(skill_gaps) / max(self.top_k_skills, 1)
        confidence = float(coverage)
        risk = self._compute_base_risk_score(confidence, user.disability_category.value)

        explanation = (
            f"Identified {len(skill_gaps)} priority upskilling targets "
            f"across {len(top_jobs)} top job opportunities."
        )
        output = AgentOutput(
            agent_type="UpskillingAgent",
            user_id=user_id,
            recommendations=skill_gaps,
            confidence=confidence,
            risk_score=risk,
            requires_human_review=self._check_governance_trigger(risk),
            explanation=explanation,
            metadata={"top_k_skills": self.top_k_skills},
        )
        self._log_decision(output)
        return output
