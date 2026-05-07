"""Employment matching agent for FedAgent-Chain.

Implements the job suitability scoring function from Section 4.7:
    S(i, r) = α·sim(u_i, j_r) + β·A(i, r) + γ·L(i, r) + δ·P(i, r)

where:
- sim(u_i, j_r): skill similarity between user i and job r
- A(i, r): accommodation compatibility score
- L(i, r): language match indicator
- P(i, r): work preference alignment

Coefficients α, β, γ, δ are loaded from configs/agents/employment_agent.yaml.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from src.agents.base_agent import AgentOutput, BaseAgent
from src.data.schema import JobProfile, UserProfile


class EmploymentAgent(BaseAgent):
    """Employment matching agent that ranks jobs by suitability score.

    Implements the suitability scoring function S(i, r) and returns a
    ranked list of job opportunities with explanations.

    Parameters
    ----------
    agent_cfg : DictConfig
        Agent configuration with fields:
        - alpha (float): Skill similarity weight.
        - beta (float): Accommodation compatibility weight.
        - gamma (float): Language match weight.
        - delta (float): Work preference alignment weight.
        - top_k (int): Number of top jobs to return.
    governance_threshold : float
        Risk score threshold for human review escalation.
    """

    def __init__(self, agent_cfg: DictConfig, governance_threshold: float = 0.7) -> None:
        super().__init__(agent_cfg, governance_threshold)
        self.alpha: float = float(agent_cfg.get("alpha", 0.40))
        self.beta: float = float(agent_cfg.get("beta", 0.25))
        self.gamma: float = float(agent_cfg.get("gamma", 0.20))
        self.delta: float = float(agent_cfg.get("delta", 0.15))
        self.top_k: int = int(agent_cfg.get("top_k", 10))

        assert abs(self.alpha + self.beta + self.gamma + self.delta - 1.0) < 1e-6, (
            "Suitability score weights must sum to 1.0"
        )

    def _compute_skill_similarity(
        self, user_skills: np.ndarray, job_skills: np.ndarray
    ) -> float:
        """Compute weighted Jaccard similarity between user and job skill vectors.

        Parameters
        ----------
        user_skills : np.ndarray
            Binary 50-dim user skill vector.
        job_skills : np.ndarray
            Binary 50-dim job required skill vector.

        Returns
        -------
        float
            Skill similarity in [0, 1].
        """
        intersection = np.sum(np.minimum(user_skills, job_skills))
        union = np.sum(np.maximum(user_skills, job_skills))
        return float(intersection / (union + 1e-8))

    def _compute_accommodation_compatibility(
        self, user_needs: np.ndarray, job_provides: np.ndarray
    ) -> float:
        """Compute accommodation coverage score A(i, r).

        Measures what fraction of the user's accommodation needs the job provides.

        Parameters
        ----------
        user_needs : np.ndarray
            Binary 20-dim accommodation needs vector.
        job_provides : np.ndarray
            Binary 20-dim accommodation provided vector.

        Returns
        -------
        float
            Accommodation coverage in [0, 1]. 1.0 = all needs met.
        """
        total_needs = float(np.sum(user_needs))
        if total_needs == 0:
            return 1.0  # No accommodation needs — fully compatible
        covered = float(np.sum(np.minimum(user_needs, job_provides)))
        return covered / total_needs

    def _compute_language_match(
        self, user_lang: str, job_lang: str, user_secondary: Optional[str] = None
    ) -> float:
        """Compute language match indicator L(i, r).

        Parameters
        ----------
        user_lang : str
            User's primary language ISO code.
        job_lang : str
            Job's required language ISO code.
        user_secondary : str, optional
            User's secondary language, if any.

        Returns
        -------
        float
            1.0 if primary language matches, 0.6 if secondary matches, 0.2 otherwise.
        """
        if user_lang == job_lang:
            return 1.0
        if user_secondary and user_secondary == job_lang:
            return 0.6
        return 0.2

    def _compute_work_preference_alignment(
        self, user_mode: str, job_mode: str, user_goal: str
    ) -> float:
        """Compute work preference alignment P(i, r).

        Parameters
        ----------
        user_mode : str
            User's preferred work mode (onsite/hybrid/remote).
        job_mode : str
            Job's work mode.
        user_goal : str
            User's employment goal (fulltime/parttime/freelance/internship).

        Returns
        -------
        float
            Preference alignment score in [0, 1].
        """
        mode_score = 1.0 if user_mode == job_mode else (0.6 if job_mode == "hybrid" else 0.3)
        return float(mode_score)

    def compute_suitability_score(
        self, user: UserProfile, job: JobProfile
    ) -> Dict[str, float]:
        """Compute the full suitability score S(i, r) for a user-job pair.

        Parameters
        ----------
        user : UserProfile
            User profile record.
        job : JobProfile
            Job profile record.

        Returns
        -------
        dict
            Dictionary with 'total_score' and per-component scores.
        """
        u_skills = np.array(user.skill_vector, dtype=float)
        j_skills = np.array(job.required_skills, dtype=float)

        skill_sim = self._compute_skill_similarity(u_skills, j_skills)
        accom = self._compute_accommodation_compatibility(
            np.array(user.accommodation_needs, dtype=float),
            np.array(job.accommodation_provided, dtype=float),
        )
        lang = self._compute_language_match(
            user.language_primary, job.language_required, user.language_secondary
        )
        pref = self._compute_work_preference_alignment(
            user.preferred_work_mode.value, job.work_mode.value, user.employment_goal.value
        )

        total = self.alpha * skill_sim + self.beta * accom + self.gamma * lang + self.delta * pref

        return {
            "total_score": float(total),
            "skill_similarity": float(skill_sim),
            "accommodation_compatibility": float(accom),
            "language_match": float(lang),
            "work_preference": float(pref),
        }

    def rank_jobs(
        self,
        user: UserProfile,
        jobs: List[JobProfile],
    ) -> List[Dict[str, Any]]:
        """Rank all available jobs for a user by suitability score.

        Parameters
        ----------
        user : UserProfile
            The target user profile.
        jobs : list of JobProfile
            Available job profiles to rank.

        Returns
        -------
        list of dict
            Top-k job recommendations sorted by suitability score (descending).
        """
        scored_jobs = []
        for job in jobs:
            scores = self.compute_suitability_score(user, job)
            scored_jobs.append(
                {
                    "job_id": job.job_id,
                    "sector": job.sector,
                    "work_mode": job.work_mode.value,
                    "language_required": job.language_required,
                    "accessibility_score": job.accessibility_score,
                    **scores,
                }
            )

        scored_jobs.sort(key=lambda x: x["total_score"], reverse=True)
        return scored_jobs[: self.top_k]

    def run(
        self,
        user_id: str,
        user: Optional[UserProfile] = None,
        jobs: Optional[List[JobProfile]] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """Execute employment matching for a user.

        Parameters
        ----------
        user_id : str
            User identifier.
        user : UserProfile
            User profile record.
        jobs : list of JobProfile
            Available job profiles.
        **kwargs : Any
            Additional keyword arguments (ignored).

        Returns
        -------
        AgentOutput
            Ranked job recommendations with suitability scores.
        """
        if user is None or jobs is None:
            raise ValueError("EmploymentAgent.run() requires 'user' and 'jobs' arguments.")

        ranked = self.rank_jobs(user, jobs)
        top_score = ranked[0]["total_score"] if ranked else 0.0
        confidence = float(top_score)

        risk_score = self._compute_base_risk_score(
            confidence=confidence,
            disability_category=user.disability_category.value,
        )
        requires_review = self._check_governance_trigger(risk_score)

        top_job = ranked[0] if ranked else {}
        explanation = (
            f"Top match job {top_job.get('job_id', 'N/A')} (sector: {top_job.get('sector', 'N/A')}) "
            f"with suitability score {top_score:.3f}. "
            f"Skill similarity: {top_job.get('skill_similarity', 0):.2f}, "
            f"Accommodation compatibility: {top_job.get('accommodation_compatibility', 0):.2f}."
        )

        output = AgentOutput(
            agent_type="EmploymentAgent",
            user_id=user_id,
            recommendations=ranked,
            confidence=confidence,
            risk_score=risk_score,
            requires_human_review=requires_review,
            explanation=explanation,
            metadata={
                "n_jobs_ranked": len(jobs),
                "top_k": self.top_k,
                "weights": {
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "gamma": self.gamma,
                    "delta": self.delta,
                },
            },
        )

        self._log_decision(output)
        return output
