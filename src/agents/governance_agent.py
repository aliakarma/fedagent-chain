"""Human-in-the-loop governance agent for FedAgent-Chain.

Classifies employment recommendations as high-risk vs. safe based on a
learned risk scoring model, and mandates human review above threshold τ.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from omegaconf import DictConfig

from src.agents.base_agent import AgentOutput, BaseAgent
from src.data.schema import DisabilityCategory


class GovernanceAgent(BaseAgent):
    """Human-in-the-loop governance agent.

    Evaluates each employment recommendation for risk indicators and
    escalates to human review whenever R_risk(d_i) > τ.

    Risk factors (additive):
    - Low confidence in top match               → base risk = 1 - confidence
    - Disability category 'multiple'            → +0.10
    - No accommodation provided for any need    → +0.15
    - Top job accessibility score < 0.3         → +0.12
    - Language mismatch (primary)               → +0.08

    Parameters
    ----------
    agent_cfg : DictConfig
        Must contain `review_threshold` (float, default 0.70).
    governance_threshold : float
        Alias for review_threshold; kept for BaseAgent API compatibility.
    """

    def __init__(
        self,
        agent_cfg: DictConfig,
        governance_threshold: float = 0.70,
    ) -> None:
        threshold = float(agent_cfg.get("review_threshold", governance_threshold))
        super().__init__(agent_cfg, governance_threshold=threshold)

    def _compute_risk_score(
        self,
        confidence: float,
        disability_category: str,
        top_recommendation: Dict[str, Any],
    ) -> float:
        """Compute R_risk(d_i) from recommendation features."""
        risk = 1.0 - confidence  # low confidence = high risk

        if disability_category == DisabilityCategory.MULTIPLE.value:
            risk = min(1.0, risk + 0.10)

        # Accommodation mismatch
        accom_compat = top_recommendation.get("accommodation_compatibility", 1.0)
        if accom_compat < 0.30:
            risk = min(1.0, risk + 0.15)

        # Accessibility score
        accessibility = top_recommendation.get("accessibility_score", 1.0)
        if accessibility < 0.30:
            risk = min(1.0, risk + 0.12)

        # Language mismatch
        lang_match = top_recommendation.get("language_match", 1.0)
        if lang_match < 0.5:
            risk = min(1.0, risk + 0.08)

        return float(np.clip(risk, 0.0, 1.0))

    def run(
        self,
        user_id: str,
        employment_output: Optional[AgentOutput] = None,
        disability_category: str = "mobility",
        **kwargs: Any,
    ) -> AgentOutput:
        """Assess risk and trigger governance review if needed.

        Parameters
        ----------
        user_id : str
            User identifier.
        employment_output : AgentOutput
            Output from EmploymentAgent (required).
        disability_category : str
            User's disability category for risk modulation.
        """
        if employment_output is None:
            raise ValueError("GovernanceAgent requires employment_output from EmploymentAgent.")

        top_rec = employment_output.recommendations[0] if employment_output.recommendations else {}
        risk    = self._compute_risk_score(
            confidence=employment_output.confidence,
            disability_category=disability_category,
            top_recommendation=top_rec,
        )
        requires_review = self._check_governance_trigger(risk)
        explanation = (
            f"Risk score {risk:.3f} {'EXCEEDS' if requires_review else 'is below'} "
            f"governance threshold τ={self.governance_threshold:.2f}. "
            f"Human review {'REQUIRED' if requires_review else 'not required'}."
        )
        output = AgentOutput(
            agent_type="GovernanceAgent",
            user_id=user_id,
            recommendations=employment_output.recommendations,
            confidence=employment_output.confidence,
            risk_score=risk,
            requires_human_review=requires_review,
            explanation=explanation,
            metadata={
                "threshold": self.governance_threshold,
                "disability_category": disability_category,
            },
        )
        self._log_decision(output)
        return output
