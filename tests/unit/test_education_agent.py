from __future__ import annotations

import numpy as np
import pytest

from src.agents.education_agent import EducationAgent
from src.data.education_catalog import (
    COMPETENCIES,
    JOB_ROLE_REQUIREMENTS,
    N_COMPETENCIES,
)


@pytest.fixture()
def agent() -> EducationAgent:
    return EducationAgent()


def test_competency_dimension_is_20() -> None:
    assert N_COMPETENCIES == 20
    assert len(COMPETENCIES) == 20
    for role, spec in JOB_ROLE_REQUIREMENTS.items():
        assert len(spec["required"]) == 20, role
        assert len(spec["importance"]) == 20, role


def test_skill_gap_is_nonnegative_and_elementwise(agent: EducationAgent) -> None:
    required = np.array([0.8] * N_COMPETENCIES)
    current = np.array([0.5] * N_COMPETENCIES)
    gap = agent.skill_gap(required, current)
    assert np.allclose(gap, 0.3)
    # No negative gaps when learner exceeds requirement
    gap2 = agent.skill_gap(np.zeros(N_COMPETENCIES), np.ones(N_COMPETENCIES))
    assert np.all(gap2 == 0.0)


def test_gap_magnitude_is_l1(agent: EducationAgent) -> None:
    gap = np.array([0.1, 0.2, 0.0] + [0.0] * (N_COMPETENCIES - 3))
    assert agent.gap_magnitude(gap) == pytest.approx(0.3)


def test_priority_weights_combine_terms(agent: EducationAgent) -> None:
    gap = np.ones(N_COMPETENCIES)
    imp = np.ones(N_COMPETENCIES)
    acc = np.ones(N_COMPETENCIES)
    rehab = np.ones(N_COMPETENCIES)
    p = agent.competency_priority(gap, imp, acc, rehab)
    # All-ones inputs => priority == sum of the four weights
    assert np.allclose(p, agent.w1 + agent.w2 + agent.w3 + agent.w4)


def test_resource_suitability_matches_formula(agent: EducationAgent) -> None:
    val = agent.resource_suitability(1.0, 1.0, 1.0, 1.0, 1.0)
    assert val == pytest.approx(
        agent.alpha_e + agent.beta_e + agent.gamma_e + agent.delta_e + agent.eta_e
    )


def test_pathway_prefers_role_aligned_resources(agent: EducationAgent) -> None:
    # A learner with zero competencies targeting data entry should get a pathway
    # whose top resources are aligned with the data-entry role.
    priority = np.ones(N_COMPETENCIES)
    pathway = agent.recommend_pathway(priority, "data_entry_assistant",
                                      accessibility_modes={"simplified_text"})
    assert len(pathway) <= agent.top_k_resources
    assert pathway, "pathway should not be empty"
    assert any(r["role_aligned"] for r in pathway)
    # Scores are sorted descending
    scores = [r["score"] for r in pathway]
    assert scores == sorted(scores, reverse=True)


def test_readiness_threshold_gating(agent: EducationAgent) -> None:
    role = "data_entry_assistant"
    b_r = JOB_ROLE_REQUIREMENTS[role]["min_readiness"]
    # High inputs => ready
    high = agent.workplace_readiness(1.0, 1.0, 1.0, 1.0)
    assert high >= b_r and agent.is_ready(high, role)
    # Low inputs => not ready -> recommend another cycle
    low = agent.workplace_readiness(0.0, 0.0, 0.0, 0.0)
    assert low < b_r and not agent.is_ready(low, role)


def test_run_ready_learner_flags_transition(agent: EducationAgent) -> None:
    competent = [0.9] * N_COMPETENCIES
    out = agent.run(
        user_id="u1", competencies=competent, target_role="data_entry_assistant",
        disability_category="cognitive", assessment_score=0.9,
        training_completion=0.9, accommodation_compatibility=0.9,
    )
    assert out.agent_type == "EducationAgent"
    assert out.metadata["ready_for_transition"] is True
    assert out.metadata["skill_gap_l1"] == pytest.approx(0.0, abs=1e-6)


def test_run_underprepared_learner_recommends_cycle(agent: EducationAgent) -> None:
    weak = [0.0] * N_COMPETENCIES
    out = agent.run(
        user_id="u2", competencies=weak, target_role="digital_accessibility_tester",
        disability_category="vision", assessment_score=0.2,
        training_completion=0.2, accommodation_compatibility=0.3,
    )
    assert out.metadata["ready_for_transition"] is False
    assert out.metadata["skill_gap_l1"] > 0.0
    assert "another adaptive learning cycle" in out.explanation


def test_run_validates_role_and_vector(agent: EducationAgent) -> None:
    with pytest.raises(ValueError):
        agent.run(user_id="x", competencies=[0.5] * N_COMPETENCIES, target_role="not_a_role")
    with pytest.raises(ValueError):
        agent.run(user_id="x", competencies=[0.5] * 5, target_role="data_entry_assistant")
    with pytest.raises(ValueError):
        agent.run(user_id="x", competencies=None, target_role="data_entry_assistant")
