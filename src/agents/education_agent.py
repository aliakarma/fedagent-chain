from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.agents.base_agent import AgentOutput, BaseAgent
from src.data.education_catalog import (
    COMPETENCIES,
    DISABILITY_ACCESSIBILITY_MODES,
    JOB_ROLE_REQUIREMENTS,
    LEARNING_RESOURCES,
    N_COMPETENCIES,
)


class EducationAgent(BaseAgent):
    """Generates accessible job-readiness pathways for learners with disabilities.

    Parameters
    ----------
    agent_cfg : DictConfig
        Optional config with:
        - priority weights ``w1..w4`` (skill gap, importance, accessibility need,
          rehabilitation relevance), default 0.40/0.30/0.20/0.10;
        - resource-suitability weights ``alpha_e, beta_e, gamma_e, delta_e, eta_e``
          (content, role, accessibility, language, preference),
          default 0.30/0.25/0.20/0.15/0.10;
        - readiness weights ``lambda1..lambda4`` (matching, assessment,
          accommodation, training), default 0.30/0.30/0.20/0.20;
        - ``top_k_resources`` (int, default 5).
    governance_threshold : float
        Risk threshold τ for human-review escalation (default 0.65, paper).
    """

    def __init__(
        self, agent_cfg: DictConfig | None = None, governance_threshold: float = 0.65
    ) -> None:
        cfg = agent_cfg if agent_cfg is not None else OmegaConf.create({})
        super().__init__(cfg, governance_threshold)
        # Priority weights w1..w4
        self.w1 = float(cfg.get("w1", 0.40))
        self.w2 = float(cfg.get("w2", 0.30))
        self.w3 = float(cfg.get("w3", 0.20))
        self.w4 = float(cfg.get("w4", 0.10))
        # Resource-suitability weights
        self.alpha_e = float(cfg.get("alpha_e", 0.30))
        self.beta_e = float(cfg.get("beta_e", 0.25))
        self.gamma_e = float(cfg.get("gamma_e", 0.20))
        self.delta_e = float(cfg.get("delta_e", 0.15))
        self.eta_e = float(cfg.get("eta_e", 0.10))
        # Readiness weights lambda1..lambda4
        self.l1 = float(cfg.get("lambda1", 0.30))
        self.l2 = float(cfg.get("lambda2", 0.30))
        self.l3 = float(cfg.get("lambda3", 0.20))
        self.l4 = float(cfg.get("lambda4", 0.20))
        self.top_k_resources = int(cfg.get("top_k_resources", 5))

    # ── Skill-gap analysis ─────────────────────────────────────────────────────
    @staticmethod
    def skill_gap(required: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Return the element-wise gap G(i,r) = max(0, q_r - s_i)."""
        return np.maximum(0.0, np.asarray(required, dtype=float) - np.asarray(current, dtype=float))

    @staticmethod
    def gap_magnitude(gap: np.ndarray) -> float:
        """Return the total skill-gap magnitude ||G(i,r)||_1."""
        return float(np.sum(np.maximum(0.0, gap)))

    def competency_priority(
        self,
        gap: np.ndarray,
        importance: np.ndarray,
        accessibility_need: np.ndarray,
        rehab_relevance: np.ndarray,
    ) -> np.ndarray:
        """Per-competency priority P_{i,r,j} = w1*G + w2*I + w3*A + w4*R."""
        gap = np.asarray(gap, dtype=float)
        importance = np.asarray(importance, dtype=float)
        accessibility_need = np.asarray(accessibility_need, dtype=float)
        rehab_relevance = np.asarray(rehab_relevance, dtype=float)
        return (
            self.w1 * gap
            + self.w2 * importance
            + self.w3 * accessibility_need
            + self.w4 * rehab_relevance
        )

    # ── Learning-resource suitability ──────────────────────────────────────────
    def resource_suitability(
        self,
        content_relevance: float,
        role_alignment: float,
        accessibility_compat: float,
        language_compat: float,
        preference: float,
    ) -> float:
        """Resource suitability E(i,l,r) = a*C + b*B + g*Acc + d*Lang + e*Pref."""
        return float(
            self.alpha_e * content_relevance
            + self.beta_e * role_alignment
            + self.gamma_e * accessibility_compat
            + self.delta_e * language_compat
            + self.eta_e * preference
        )

    def recommend_pathway(
        self,
        priority: np.ndarray,
        target_role: str,
        accessibility_modes: set,
        language: str = "multi",
    ) -> list[dict[str, Any]]:
        """Select the top-k accessible learning resources for the target role.

        Each candidate resource is scored with ``resource_suitability``; the
        content-relevance term ``C`` is the priority mass the resource covers,
        the role term ``B`` is whether the resource aligns with the target role,
        ``Acc`` is the fraction of the resource's modes matching the learner's
        accessibility needs, ``Lang`` is language compatibility, and ``Pref`` a
        mild ordering preference.
        """
        priority = np.asarray(priority, dtype=float)
        total_priority = float(np.sum(priority)) or 1.0
        scored: list[dict[str, Any]] = []
        for res in LEARNING_RESOURCES:
            idx = [COMPETENCIES.index(c) for c in res["competencies"] if c in COMPETENCIES]
            content = float(np.sum(priority[idx])) / total_priority if idx else 0.0
            role = 1.0 if target_role in res["roles"] else 0.0
            modes = res["accessibility_modes"]
            acc = (
                len(modes & accessibility_modes) / len(accessibility_modes)
                if accessibility_modes
                else 1.0
            )
            lang = 1.0 if res["language"] in ("multi", language) else 0.0
            pref = 1.0  # neutral preference; learner-specific tuning is future work
            score = self.resource_suitability(content, role, acc, lang, pref)
            scored.append(
                {
                    "resource_id": res["resource_id"],
                    "title": res["title"],
                    "score": round(score, 4),
                    "covers": res["competencies"],
                    "role_aligned": bool(role),
                }
            )
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[: self.top_k_resources]

    # ── Workplace-readiness evaluation ─────────────────────────────────────────
    def workplace_readiness(
        self,
        matching: float,
        assessment: float,
        accommodation: float,
        training: float,
    ) -> float:
        """Readiness W(i,r) = l1*M + l2*AScore + l3*CScore + l4*TScore."""
        return float(
            self.l1 * matching + self.l2 * assessment + self.l3 * accommodation + self.l4 * training
        )

    def is_ready(self, readiness: float, target_role: str) -> bool:
        """Return True iff W(i,r) >= b_r for the target role."""
        b_r = JOB_ROLE_REQUIREMENTS[target_role]["min_readiness"]
        return readiness >= b_r

    # ── Main entry point ───────────────────────────────────────────────────────
    def run(
        self,
        user_id: str,
        competencies: list[float] | None = None,
        target_role: str | None = None,
        disability_category: str = "cognitive",
        language: str = "multi",
        assessment_score: float = 0.7,
        training_completion: float = 0.7,
        accommodation_compatibility: float = 0.8,
        rehab_relevance: list[float] | None = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """Generate an accessible job-readiness pathway and readiness decision.

        Parameters
        ----------
        competencies : list of float
            Learner competency vector ``s_i`` (length 20, levels in [0, 1]).
        target_role : str
            One of :data:`JOB_ROLE_REQUIREMENTS` keys.
        disability_category : str
            Drives the accessibility-mode profile and accessibility-need weighting.
        assessment_score, training_completion, accommodation_compatibility : float
            Inputs to the workplace-readiness score (AScore, TScore, CScore).
        """
        if target_role is None or target_role not in JOB_ROLE_REQUIREMENTS:
            raise ValueError(
                f"target_role must be one of {list(JOB_ROLE_REQUIREMENTS)}; got {target_role!r}"
            )
        if competencies is None:
            raise ValueError(
                "EducationAgent requires the learner competency vector 'competencies'."
            )
        s_i = np.asarray(competencies, dtype=float)
        if s_i.shape[0] != N_COMPETENCIES:
            raise ValueError(f"competencies must have length {N_COMPETENCIES}, got {s_i.shape[0]}")

        role = JOB_ROLE_REQUIREMENTS[target_role]
        q_r = np.asarray(role["required"], dtype=float)
        importance = np.asarray(role["importance"], dtype=float)

        gap = self.skill_gap(q_r, s_i)
        gap_l1 = self.gap_magnitude(gap)

        # Accessibility-need weighting: competencies tied to the learner's assistive
        # modes (e.g. screen_reader_use for vision) get a higher accessibility need.
        access_need = np.zeros(N_COMPETENCIES)
        modes = DISABILITY_ACCESSIBILITY_MODES.get(disability_category, set())
        if "screen_reader" in modes or "keyboard_only" in modes:
            for c in ("screen_reader_use", "keyboard_navigation", "assistive_tech_readiness"):
                access_need[COMPETENCIES.index(c)] = 1.0
        if "simplified_text" in modes or "structured_tasks" in modes:
            for c in ("task_accuracy", "attention_to_detail", "workplace_behavior"):
                access_need[COMPETENCIES.index(c)] = 1.0

        rehab = (
            np.asarray(rehab_relevance, dtype=float)
            if rehab_relevance is not None
            else np.zeros(N_COMPETENCIES)
        )

        priority = self.competency_priority(gap, importance, access_need, rehab)
        pathway = self.recommend_pathway(priority, target_role, modes, language)

        # Matching score M: how close the learner already is to the requirement.
        matching = float(1.0 - gap_l1 / max(float(np.sum(q_r)), 1.0))
        matching = float(np.clip(matching, 0.0, 1.0))
        readiness = self.workplace_readiness(
            matching, assessment_score, accommodation_compatibility, training_completion
        )
        ready = self.is_ready(readiness, target_role)
        b_r = role["min_readiness"]

        # Risk: low readiness or many unmet high-importance gaps raise risk.
        confidence = float(np.clip(readiness, 0.0, 1.0))
        risk = self._compute_base_risk_score(confidence, disability_category)

        top_gaps = [
            {
                "competency": COMPETENCIES[j],
                "gap": round(float(gap[j]), 3),
                "priority": round(float(priority[j]), 3),
            }
            for j in np.argsort(priority)[::-1]
            if gap[j] > 0
        ][: self.top_k_resources]

        explanation = (
            f"Target role '{target_role}': skill-gap magnitude ||G||_1={gap_l1:.2f}; "
            f"readiness W={readiness:.3f} {'>=' if ready else '<'} b_r={b_r:.2f} -> "
            f"{'ready for human-reviewed transition' if ready else 'recommend another adaptive learning cycle'}."
        )

        output = AgentOutput(
            agent_type="EducationAgent",
            user_id=user_id,
            recommendations=pathway,
            confidence=confidence,
            risk_score=risk,
            requires_human_review=self._check_governance_trigger(risk),
            explanation=explanation,
            metadata={
                "target_role": target_role,
                "skill_gap_l1": round(gap_l1, 4),
                "readiness": round(readiness, 4),
                "readiness_threshold": b_r,
                "ready_for_transition": ready,
                "top_priority_gaps": top_gaps,
                "disability_category": disability_category,
            },
        )
        self._log_decision(output)
        return output
