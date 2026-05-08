"""Multilingual communication support agent for FedAgent-Chain."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig

from src.agents.base_agent import AgentOutput, BaseAgent
from src.data.schema import UserProfile

# Supported language pairs and coverage quality scores
# (In production, replace with actual multilingual model evaluation)
_LANGUAGE_SUPPORT: Dict[str, float] = {
    "ar": 0.92, "en": 0.98, "zh": 0.91,
    "fr": 0.93, "de": 0.90, "es": 0.94,
    "ur": 0.83, "tl": 0.78, "yue": 0.81,
}


class MultilingualCommunicationAgent(BaseAgent):
    """Assesses and supports multilingual communication for employment.

    Evaluates language adequacy between user's language profile and the
    job's required language, and recommends communication support resources.

    Parameters
    ----------
    agent_cfg : DictConfig
        Config with `supported_languages` (list, optional).
    """

    def run(
        self,
        user_id: str,
        user: Optional[UserProfile] = None,
        job_language: str = "en",
        **kwargs: Any,
    ) -> AgentOutput:
        if user is None:
            raise ValueError("MultilingualCommunicationAgent requires 'user'.")

        primary_lang   = user.language_primary
        secondary_lang = user.language_secondary

        # Language adequacy score
        primary_match   = (primary_lang == job_language)
        secondary_match = (secondary_lang == job_language) if secondary_lang else False

        if primary_match:
            adequacy   = _LANGUAGE_SUPPORT.get(primary_lang, 0.75)
            support_needed = False
        elif secondary_match:
            adequacy   = _LANGUAGE_SUPPORT.get(secondary_lang, 0.75) * 0.85
            support_needed = True
        else:
            adequacy   = 0.35
            support_needed = True

        recommendations = []
        if support_needed:
            recommendations = [
                {
                    "type":     "language_bridge",
                    "resource": f"Translation support: {primary_lang} ↔ {job_language}",
                    "adequacy": round(adequacy, 3),
                },
                {
                    "type":     "language_course",
                    "resource": f"Language training programme for {job_language}",
                    "adequacy": round(adequacy, 3),
                },
            ]

        confidence = float(adequacy)
        risk       = self._compute_base_risk_score(confidence, user.disability_category.value)

        explanation = (
            f"Language adequacy score: {adequacy:.3f}. "
            f"Communication support {'recommended' if support_needed else 'not required'}."
        )
        output = AgentOutput(
            agent_type="MultilingualCommunicationAgent",
            user_id=user_id,
            recommendations=recommendations,
            confidence=confidence,
            risk_score=risk,
            requires_human_review=self._check_governance_trigger(risk),
            explanation=explanation,
            metadata={
                "primary_language": primary_lang,
                "job_language": job_language,
                "adequacy": adequacy,
                "support_needed": support_needed,
            },
        )
        self._log_decision(output)
        return output
