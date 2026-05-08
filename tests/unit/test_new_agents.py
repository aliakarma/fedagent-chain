"""Unit tests for Phase 4 agents."""
import pytest
from omegaconf import OmegaConf
from src.agents.governance_agent import GovernanceAgent
from src.agents.upskilling_agent import UpskillingAgent
from src.agents.accommodation_agent import AccommodationAgent
from src.agents.multilingual_agent import MultilingualCommunicationAgent
from src.agents.base_agent import AgentOutput
from src.data.synthetic_generator import generate_user_profiles, generate_job_profiles
from src.utils.seed_utils import get_rng


@pytest.fixture
def base_cfg():
    return OmegaConf.create({
        "alpha": 0.40, "beta": 0.25, "gamma": 0.20, "delta": 0.15,
        "top_k": 5, "top_k_skills": 5, "review_threshold": 0.70,
    })

@pytest.fixture
def users_jobs():
    rng = get_rng(42)
    users = generate_user_profiles("saudi_arabia", 5, rng)
    jobs  = generate_job_profiles("saudi_arabia", 20, rng)
    return users, jobs


class TestGovernanceAgent:
    def test_high_risk_triggers_review(self, base_cfg, users_jobs):
        from src.agents.employment_agent import EmploymentAgent
        users, jobs = users_jobs
        emp   = EmploymentAgent(base_cfg, governance_threshold=0.999)
        gov   = GovernanceAgent(OmegaConf.create({"review_threshold": 0.01}))
        emp_out = emp.run(users[0].user_id, user=users[0], jobs=jobs)
        gov_out = gov.run(users[0].user_id, employment_output=emp_out,
                          disability_category=users[0].disability_category.value)
        assert gov_out.requires_human_review is True

    def test_output_is_agent_output(self, base_cfg, users_jobs):
        from src.agents.employment_agent import EmploymentAgent
        users, jobs = users_jobs
        emp = EmploymentAgent(base_cfg)
        gov = GovernanceAgent(OmegaConf.create({"review_threshold": 0.70}))
        emp_out = emp.run(users[0].user_id, user=users[0], jobs=jobs)
        gov_out = gov.run(users[0].user_id, employment_output=emp_out,
                          disability_category=users[0].disability_category.value)
        assert isinstance(gov_out, AgentOutput)
        assert gov_out.agent_type == "GovernanceAgent"
        assert 0.0 <= gov_out.risk_score <= 1.0


class TestUpskillingAgent:
    def test_returns_skill_gaps(self, base_cfg, users_jobs):
        users, jobs = users_jobs
        agent = UpskillingAgent(base_cfg)
        out   = agent.run(users[0].user_id, user=users[0], top_jobs=jobs[:5])
        assert isinstance(out, AgentOutput)
        assert len(out.recommendations) <= base_cfg.top_k_skills

    def test_skills_have_correct_keys(self, base_cfg, users_jobs):
        users, jobs = users_jobs
        agent = UpskillingAgent(base_cfg)
        out   = agent.run(users[0].user_id, user=users[0], top_jobs=jobs[:10])
        for rec in out.recommendations:
            assert "skill_index" in rec
            assert "frequency" in rec


class TestAccommodationAgent:
    def test_perfect_coverage_gives_full_confidence(self, base_cfg, users_jobs):
        import numpy as np
        from src.data.schema import (
            UserProfile, JobProfile, NodeID, DisabilityCategory,
            WorkMode, EmploymentGoal, EducationLevel
        )
        user = UserProfile(
            user_id="u1", node_id=NodeID.SAUDI_ARABIA,
            skill_vector=[1,0]*25, education_level=EducationLevel.UNDERGRADUATE,
            disability_category=DisabilityCategory.MOBILITY,
            accommodation_needs=[1]*20,           # all needs
            language_primary="ar", preferred_work_mode=WorkMode.HYBRID,
            employment_goal=EmploymentGoal.FULLTIME, consent_given=True,
        )
        job = JobProfile(
            job_id="j1", node_id=NodeID.SAUDI_ARABIA,
            required_skills=[1,0]*25, accessibility_score=0.9,
            work_mode=WorkMode.HYBRID, language_required="ar",
            education_minimum=EducationLevel.UNDERGRADUATE,
            accommodation_provided=[1]*20,        # all provided
            sector="technology",
        )
        agent = AccommodationAgent(base_cfg)
        out   = agent.run("u1", user=user, job=job)
        assert out.confidence == pytest.approx(1.0), "All needs met → confidence 1.0"
        assert len(out.recommendations) == 0


class TestMultilingualAgent:
    def test_primary_language_match_high_adequacy(self, base_cfg, users_jobs):
        users, _ = users_jobs
        agent = MultilingualCommunicationAgent(base_cfg)
        # Find a user with primary language "ar" and test against "ar" job
        for user in users:
            if user.language_primary == "ar":
                out = agent.run(user.user_id, user=user, job_language="ar")
                assert out.confidence > 0.80
                break

    def test_no_match_low_adequacy(self, base_cfg, users_jobs):
        users, _ = users_jobs
        agent = MultilingualCommunicationAgent(base_cfg)
        out   = agent.run(users[0].user_id, user=users[0], job_language="xx")
        assert out.confidence < 0.60
