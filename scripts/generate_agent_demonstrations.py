#!/usr/bin/env python3
"""Generate qualitative agentic demonstrations for Phase 5.

Simulates 3 specific scenarios and records the multi-agent orchestration
outputs into markdown files for the documentation.
"""

from __future__ import annotations
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd

from src.agents.employment_agent import EmploymentAgent
from src.agents.governance_agent import GovernanceAgent
from src.agents.upskilling_agent import UpskillingAgent
from src.agents.accommodation_agent import AccommodationAgent
from src.agents.multilingual_agent import MultilingualCommunicationAgent
from src.data.schema import UserProfile, JobProfile, DisabilityCategory, WorkMode, EducationLevel, EmploymentGoal

def create_scenarios():
    # Scenario 1: Arabic-speaking, visually impaired, seeking Data Analyst role
    s1_user = UserProfile(
        user_id="user_v1_arabic",
        node_id="saudi_arabia",
        disability_category=DisabilityCategory.VISION,
        language_primary="ar",
        preferred_work_mode=WorkMode.REMOTE,
        education_level=EducationLevel.UNDERGRADUATE,
        employment_goal=EmploymentGoal.FULLTIME,
        skill_vector=[1, 1, 0, 0, 1] + [0]*45, # High tech skills (binary)
        accommodation_needs=[1, 1] + [0]*18,
        consent_given=True
    )
    s1_job = JobProfile(
        job_id="job_data_analyst",
        node_id="saudi_arabia",
        accessibility_score=0.9,
        language_required="en",
        work_mode=WorkMode.HYBRID,
        required_skills=[1, 1, 0, 0, 1] + [0]*45,
        education_minimum=EducationLevel.UNDERGRADUATE,
        accommodation_provided=[1, 1] + [0]*18,
        sector="technology"
    )

    # Scenario 2: Mobility impaired, English, prefers Remote, needs Upskilling
    s2_user = UserProfile(
        user_id="user_m1_remote",
        node_id="united_states",
        disability_category=DisabilityCategory.MOBILITY,
        language_primary="en",
        preferred_work_mode=WorkMode.REMOTE,
        education_level=EducationLevel.UNDERGRADUATE,
        employment_goal=EmploymentGoal.FULLTIME,
        skill_vector=[0]*50, # Low skills, needs upskilling
        accommodation_needs=[1, 0] + [0]*18,
        consent_given=True
    )
    s2_job = JobProfile(
        job_id="job_admin_remote",
        node_id="united_states",
        accessibility_score=0.95,
        language_required="en",
        work_mode=WorkMode.REMOTE,
        required_skills=[0]*10 + [1, 1, 1] + [0]*37, # Does NOT match user skills
        education_minimum=EducationLevel.UNDERGRADUATE,
        accommodation_provided=[1, 0] + [0]*18,
        sector="finance"
    )

    # Scenario 3: High-Risk (Multiple disabilities, very low confidence)
    s3_user = UserProfile(
        user_id="user_risk_high",
        node_id="europe",
        disability_category=DisabilityCategory.MULTIPLE,
        language_primary="en",
        preferred_work_mode=WorkMode.ONSITE,
        education_level=EducationLevel.NONE,
        employment_goal=EmploymentGoal.FULLTIME,
        skill_vector=[0]*50,
        accommodation_needs=[1]*20,
        consent_given=True
    )
    s3_job = JobProfile(
        job_id="job_manual_labor",
        node_id="europe",
        accessibility_score=0.2, # Unfriendly
        language_required="en",
        work_mode=WorkMode.ONSITE,
        required_skills=[0]*40 + [1, 1] + [0]*8,
        education_minimum=EducationLevel.NONE,
        accommodation_provided=[0]*20,
        sector="manufacturing"
    )

    return [(s1_user, s1_job, "Scenario 1: Arabic/Visual Accessibility"),
            (s2_user, s2_job, "Scenario 2: Remote Work & Upskilling"),
            (s3_user, s3_job, "Scenario 3: Governance Risk Detection")]

def run_agents(user, job, scenario_name):
    cfg = OmegaConf.create({
        "alpha": 0.40, "beta": 0.25, "gamma": 0.20, "delta": 0.15, "top_k": 5,
        "top_k_skills": 3, "review_threshold": 0.40
    })
    
    emp_agent = EmploymentAgent(cfg)
    ups_agent = UpskillingAgent(cfg)
    acc_agent = AccommodationAgent(cfg)
    lang_agent = MultilingualCommunicationAgent(cfg)
    gov_agent = GovernanceAgent(cfg)

    # 1. Employment Matching
    emp_out = emp_agent.run(user_id=user.user_id, user=user, jobs=[job])
    
    # 2. Upskilling
    ups_out = ups_agent.run(user_id=user.user_id, user=user, top_jobs=[job])
    
    # 3. Accommodation
    acc_out = acc_agent.run(user_id=user.user_id, user=user, job=job)
    
    # 4. Multilingual
    lang_out = lang_agent.run(user_id=user.user_id, user=user, job_language=job.language_required)
    
    # 5. Governance
    gov_out = gov_agent.run(
        user_id=user.user_id,
        employment_output=emp_out,
        disability_category=user.disability_category.value
    )

    return {
        "scenario": scenario_name,
        "user": user,
        "job": job,
        "emp": emp_out,
        "ups": ups_out,
        "acc": acc_out,
        "lang": lang_out,
        "gov": gov_out
    }

def format_as_markdown(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Agentic AI Demonstration: {results['scenario']}\n\n")
        
        f.write("## 👤 User Profile\n")
        f.write(f"- **ID**: {results['user'].user_id}\n")
        f.write(f"- **Disability**: {results['user'].disability_category.value}\n")
        f.write(f"- **Language**: {results['user'].language_primary}\n")
        f.write(f"- **Work Mode**: {results['user'].preferred_work_mode.value}\n\n")

        f.write("## 💼 Target Job\n")
        f.write(f"- **ID**: {results['job'].job_id}\n")
        f.write(f"- **Work Mode**: {results['job'].work_mode.value}\n")
        f.write(f"- **Friendly Score**: {results['job'].accessibility_score}\n\n")

        f.write("## 🤖 Agent Outputs\n\n")
        
        f.write("### 🤝 Employment Agent\n")
        f.write(f"- **Confidence**: {results['emp'].confidence:.2f}\n")
        f.write(f"- **Reasoning**: {results['emp'].metadata.get('reasoning', 'N/A')}\n\n")

        f.write("### 📚 Upskilling Agent\n")
        f.write(f"- **Recommended Courses**: {', '.join(results['ups'].recommendations[0].get('suggested_courses', [])) if results['ups'].recommendations else 'None'}\n\n")

        f.write("### 🏠 Accommodation Agent\n")
        f.write(f"- **Strategy**: {results['acc'].recommendations[0].get('strategy', 'N/A') if results['acc'].recommendations else 'N/A'}\n")
        f.write(f"- **Cost Est**: {results['acc'].recommendations[0].get('estimated_cost', 'N/A') if results['acc'].recommendations else 'N/A'}\n\n")

        f.write("### 🌐 Multilingual Agent\n")
        f.write(f"- **Communication Plan**: {results['lang'].metadata.get('plan', 'N/A')}\n\n")

        f.write("### ⚖️ Governance Agent\n")
        f.write(f"- **Risk Score**: {results['gov'].risk_score:.2f}\n")
        f.write(f"- **Status**: {'🚩 FLAG FOR HUMAN REVIEW' if results['gov'].risk_score > 0.4 else '✅ APPROVED'}\n")
        f.write(f"- **Audit**: {results['gov'].metadata.get('audit_comment', 'N/A')}\n")

def main():
    scenarios = create_scenarios()
    results_dir = Path("experiments/results/demos")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_summaries = []
    
    for user, job, name in scenarios:
        res = run_agents(user, job, name)
        safe_name = name.lower().replace(":", "").replace(" ", "_").replace("/", "_")
        format_as_markdown(res, results_dir / f"{safe_name}.md")
        all_summaries.append(res)
        
    print(f"\n[OK] Agent demonstrations generated in {results_dir}")

if __name__ == "__main__":
    main()
