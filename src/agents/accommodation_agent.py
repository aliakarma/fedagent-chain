"""Workplace accommodation recommendation agent for FedAgent-Chain."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from omegaconf import DictConfig

from src.agents.base_agent import AgentOutput, BaseAgent
from src.data.schema import JobProfile, UserProfile

# WHO-ICF aligned accommodation category labels (indices 0–19)
ACCOMMODATION_LABELS = [
    "screen_reader", "wheelchair_access", "sign_language",
    "flexible_hours", "remote_option", "large_print",
    "hearing_loop", "ergonomic_equipment", "transport_support",
    "mental_health_support", "sensory_room", "quiet_workspace",
    "job_coaching", "adapted_keyboard", "voice_recognition",
    "braille_materials", "service_animal_policy", "interpreter",
    "medication_schedule", "physical_therapy_access",
]


class AccommodationAgent(BaseAgent):
    """Recommends specific workplace accommodations based on unmet needs.

    For each user need not covered by the target job, recommends the
    corresponding accommodation with a priority score.
    """

    def run(
        self,
        user_id: str,
        user: Optional[UserProfile] = None,
        job: Optional[JobProfile] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        if user is None or job is None:
            raise ValueError("AccommodationAgent requires 'user' and 'job'.")

        u_needs  = np.array(user.accommodation_needs,    dtype=int)
        j_covers = np.array(job.accommodation_provided,  dtype=int)
        unmet    = np.maximum(0, u_needs - j_covers)

        recommendations = []
        for idx in np.where(unmet == 1)[0]:
            label = (
                ACCOMMODATION_LABELS[idx]
                if idx < len(ACCOMMODATION_LABELS) else f"accommodation_{idx}"
            )
            recommendations.append({
                "accommodation_index": int(idx),
                "label":  label,
                "unmet":  True,
                "priority": int(u_needs[idx]),
            })

        n_unmet     = len(recommendations)
        n_needs     = int(np.sum(u_needs))
        coverage    = 1.0 - (n_unmet / max(n_needs, 1))
        confidence  = float(coverage)
        risk        = self._compute_base_risk_score(
            confidence, user.disability_category.value
        )

        explanation = (
            f"{n_unmet} accommodation(s) unmet out of {n_needs} needs. "
            f"Coverage: {coverage:.1%}."
        )
        output = AgentOutput(
            agent_type="AccommodationAgent",
            user_id=user_id,
            recommendations=recommendations,
            confidence=confidence,
            risk_score=risk,
            requires_human_review=self._check_governance_trigger(risk),
            explanation=explanation,
            metadata={"n_needs": n_needs, "n_unmet": n_unmet, "coverage": coverage},
        )
        self._log_decision(output)
        return output
