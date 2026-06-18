"""Abstract base class for all FedAgent-Chain agentic AI services.

All agents operate under policy constraints and cannot independently make
final employment decisions for high-risk cases. Every decision is logged
to the blockchain audit trail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from src.utils.logging_utils import get_logger


@dataclass
class AgentOutput:
    """Standard output container for all agent responses.

    Attributes
    ----------
    agent_type : str
        Identifier of the agent that produced this output.
    user_id : str
        The user this output is generated for.
    recommendations : list
        Ordered list of recommendation items.
    confidence : float
        Confidence score in [0, 1] for the top recommendation.
    risk_score : float
        Risk score in [0, 1] indicating need for human review.
    requires_human_review : bool
        Whether the governance agent should escalate this to a human.
    explanation : str
        Natural language explanation of the recommendation.
    metadata : dict
        Additional agent-specific metadata.
    """

    agent_type: str
    user_id: str
    recommendations: list[dict[str, Any]]
    confidence: float
    risk_score: float
    requires_human_review: bool
    explanation: str
    metadata: dict[str, Any]


class BaseAgent(ABC):
    """Abstract base class for all FedAgent-Chain agentic AI services.

    Enforces:
    - Policy constraint validation before any recommendation
    - Mandatory governance review triggering for high-risk cases
    - Structured audit logging for every decision
    - Human-in-the-loop escalation at configurable risk threshold τ

    Parameters
    ----------
    agent_cfg : DictConfig
        Agent-specific configuration parameters.
    governance_threshold : float
        Risk score threshold τ above which human review is mandatory.
        Default: 0.7 (per configs/agents/governance_agent.yaml).
    """

    def __init__(
        self,
        agent_cfg: DictConfig,
        governance_threshold: float = 0.7,
    ) -> None:
        self.cfg = agent_cfg
        self.governance_threshold = governance_threshold
        self.logger = get_logger(self.__class__.__name__)
        self._decision_count = 0

    @abstractmethod
    def run(self, user_id: str, **kwargs: Any) -> AgentOutput:
        """Execute the agent's core task for a given user.

        Parameters
        ----------
        user_id : str
            The user to generate recommendations for.
        **kwargs : Any
            Agent-specific input parameters.

        Returns
        -------
        AgentOutput
            Structured agent output with recommendations and metadata.
        """
        ...

    def _check_governance_trigger(self, risk_score: float) -> bool:
        """Determine whether human governance review is required.

        Triggers if R_risk(d_i) > τ, as specified in Section 4.9
        of the FedAgent-Chain paper.

        Parameters
        ----------
        risk_score : float
            Computed risk score for the current recommendation.

        Returns
        -------
        bool
            True if human review must be triggered, False otherwise.
        """
        triggered = risk_score > self.governance_threshold
        if triggered:
            self.logger.warning(
                "Governance review triggered",
                risk_score=round(risk_score, 4),
                threshold=self.governance_threshold,
            )
        return triggered

    def _compute_base_risk_score(
        self,
        confidence: float,
        disability_category: str | None = None,
        is_high_stakes: bool = False,
    ) -> float:
        """Compute a base risk score for governance triggering.

        Risk is higher when confidence is lower and when the case involves
        high-stakes employment decisions or vulnerable population subgroups.

        Parameters
        ----------
        confidence : float
            Agent's confidence in its top recommendation.
        disability_category : str, optional
            Disability category, used to apply additional caution.
        is_high_stakes : bool
            Whether this involves a high-stakes decision context.

        Returns
        -------
        float
            Risk score in [0, 1].
        """
        base_risk = 1.0 - confidence
        if is_high_stakes:
            base_risk = min(1.0, base_risk + 0.15)
        if disability_category == "multiple":
            base_risk = min(1.0, base_risk + 0.10)
        return float(base_risk)

    def _log_decision(self, output: AgentOutput) -> None:
        """Log the agent decision to the structured audit trail.

        Parameters
        ----------
        output : AgentOutput
            The agent's output to be logged.
        """
        self._decision_count += 1
        self.logger.info(
            "Agent decision logged",
            agent_type=output.agent_type,
            decision_count=self._decision_count,
            risk_score=round(output.risk_score, 4),
            requires_human_review=output.requires_human_review,
            confidence=round(output.confidence, 4),
            n_recommendations=len(output.recommendations),
        )

    @property
    def total_decisions(self) -> int:
        """Return total number of decisions made by this agent instance."""
        return self._decision_count
