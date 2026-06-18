"""Unit tests for agentic AI services."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.agents.base_agent import AgentOutput
from src.agents.employment_agent import EmploymentAgent
from src.data.synthetic_generator import generate_job_profiles, generate_user_profiles
from src.utils.seed_utils import get_rng


@pytest.fixture
def agent_cfg():
    return OmegaConf.create(
        {
            "alpha": 0.40,
            "beta": 0.25,
            "gamma": 0.20,
            "delta": 0.15,
            "top_k": 5,
        }
    )


@pytest.fixture
def users_and_jobs():
    rng = get_rng(42)
    users = generate_user_profiles("saudi_arabia", 5, rng)
    jobs = generate_job_profiles("saudi_arabia", 20, rng)
    return users, jobs


class TestEmploymentAgent:

    def test_run_returns_agent_output(self, agent_cfg, users_and_jobs):
        users, jobs = users_and_jobs
        agent = EmploymentAgent(agent_cfg, governance_threshold=0.7)
        output = agent.run(user_id=users[0].user_id, user=users[0], jobs=jobs)
        assert isinstance(output, AgentOutput)
        assert output.agent_type == "EmploymentAgent"

    def test_top_k_recommendations_returned(self, agent_cfg, users_and_jobs):
        users, jobs = users_and_jobs
        agent = EmploymentAgent(agent_cfg, governance_threshold=0.7)
        output = agent.run(user_id=users[0].user_id, user=users[0], jobs=jobs)
        assert len(output.recommendations) <= agent.top_k

    def test_recommendations_sorted_by_score(self, agent_cfg, users_and_jobs):
        users, jobs = users_and_jobs
        agent = EmploymentAgent(agent_cfg, governance_threshold=0.7)
        output = agent.run(user_id=users[0].user_id, user=users[0], jobs=jobs)
        scores = [r["total_score"] for r in output.recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(self, agent_cfg, users_and_jobs):
        users, jobs = users_and_jobs
        agent = EmploymentAgent(agent_cfg, governance_threshold=0.7)
        output = agent.run(user_id=users[0].user_id, user=users[0], jobs=jobs)
        for rec in output.recommendations:
            assert 0.0 <= rec["total_score"] <= 1.0

    def test_weights_sum_to_one(self, agent_cfg):
        agent = EmploymentAgent(agent_cfg)
        assert abs(agent.alpha + agent.beta + agent.gamma + agent.delta - 1.0) < 1e-6

    def test_governance_trigger_high_risk(self, agent_cfg, users_and_jobs):
        """Low confidence should trigger governance review."""
        users, jobs = users_and_jobs
        agent = EmploymentAgent(agent_cfg, governance_threshold=0.01)  # very low threshold
        output = agent.run(user_id=users[0].user_id, user=users[0], jobs=jobs)
        assert output.requires_human_review is True

    def test_governance_not_triggered_low_risk(self, agent_cfg, users_and_jobs):
        """High-confidence match should not trigger governance with high threshold."""
        users, jobs = users_and_jobs
        agent = EmploymentAgent(agent_cfg, governance_threshold=0.999)  # very high threshold
        output = agent.run(user_id=users[0].user_id, user=users[0], jobs=jobs)
        assert output.requires_human_review is False

    def test_missing_user_raises(self, agent_cfg):
        agent = EmploymentAgent(agent_cfg)
        with pytest.raises((ValueError, TypeError)):
            agent.run(user_id="u1", user=None, jobs=[])

    def test_decision_counter_increments(self, agent_cfg, users_and_jobs):
        users, jobs = users_and_jobs
        agent = EmploymentAgent(agent_cfg)
        assert agent.total_decisions == 0
        agent.run(user_id=users[0].user_id, user=users[0], jobs=jobs)
        assert agent.total_decisions == 1
        agent.run(user_id=users[1].user_id, user=users[1], jobs=jobs)
        assert agent.total_decisions == 2

    def test_language_match_full_score(self, agent_cfg):
        agent = EmploymentAgent(agent_cfg)
        score = agent._compute_language_match("ar", "ar")
        assert score == pytest.approx(1.0)

    def test_language_mismatch_low_score(self, agent_cfg):
        agent = EmploymentAgent(agent_cfg)
        score = agent._compute_language_match("ar", "zh")
        assert score == pytest.approx(0.2)

    def test_language_secondary_match(self, agent_cfg):
        agent = EmploymentAgent(agent_cfg)
        score = agent._compute_language_match("ar", "en", user_secondary="en")
        assert score == pytest.approx(0.6)

    def test_perfect_accommodation_coverage(self, agent_cfg):
        import numpy as np

        agent = EmploymentAgent(agent_cfg)
        needs = np.array([1, 1, 0, 1, 0] * 4, dtype=float)
        provided = np.array([1, 1, 1, 1, 1] * 4, dtype=float)
        score = agent._compute_accommodation_compatibility(needs, provided)
        assert score == pytest.approx(1.0)

    def test_zero_accommodation_needs_full_score(self, agent_cfg):
        import numpy as np

        agent = EmploymentAgent(agent_cfg)
        needs = np.zeros(20, dtype=float)
        provided = np.zeros(20, dtype=float)
        score = agent._compute_accommodation_compatibility(needs, provided)
        assert score == pytest.approx(1.0)
