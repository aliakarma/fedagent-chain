from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.education_agent import EducationAgent
from src.data.education_catalog import (
    COMPETENCIES,
    DISABILITY_ADAPTATIONS,
    N_COMPETENCIES,
)
from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("generate_education_demos")


def _competency_vector(**levels: float) -> list[float]:
    v = [0.0] * N_COMPETENCIES
    for name, lvl in levels.items():
        v[COMPETENCIES.index(name)] = lvl
    return v


# Four illustrative learners (competency levels reflect partial readiness with
# role-specific gaps, matching the paper's narrative).
SCENARIOS = [
    {
        "slug": "education_pathway_1_cognitive_data_entry",
        "title": "Mild Cognitive Impairment -> Data-Entry Assistant",
        "disability_label": "mild cognitive impairment",
        "disability_category": "cognitive",
        "target_role": "data_entry_assistant",
        "language": "en",
        "competencies": _competency_vector(
            digital_literacy=0.5, data_entry=0.4, spreadsheet_skills=0.3,
            task_accuracy=0.45, attention_to_detail=0.4, time_management=0.5,
            assessment_performance=0.5),
        "assessment_score": 0.55, "training_completion": 0.5,
        "accommodation_compatibility": 0.75,
        "accommodations": ["task checklists", "flexible task duration",
                           "supervisor verification at defined intervals"],
    },
    {
        "slug": "education_pathway_2_hearing_customer_support",
        "title": "Hearing Impairment -> Remote Customer Support Officer",
        "disability_label": "hearing impairment",
        "disability_category": "hearing",
        "target_role": "remote_customer_support_officer",
        "language": "en",
        "competencies": _competency_vector(
            digital_literacy=0.6, language_proficiency=0.6, written_communication=0.5,
            customer_service=0.4, ticket_handling=0.3, workplace_behavior=0.6,
            time_management=0.55, assessment_performance=0.55),
        "assessment_score": 0.6, "training_completion": 0.55,
        "accommodation_compatibility": 0.8,
        "accommodations": ["captioned meetings", "text-based escalation channels",
                           "platforms with visual notifications"],
    },
    {
        "slug": "education_pathway_3_vision_accessibility_tester",
        "title": "Visual Impairment -> Digital Accessibility Tester",
        "disability_label": "visual impairment",
        "disability_category": "vision",
        "target_role": "digital_accessibility_tester",
        "language": "en",
        "competencies": _competency_vector(
            digital_literacy=0.6, accessibility_testing=0.4, screen_reader_use=0.6,
            keyboard_navigation=0.55, defect_reporting=0.35, attention_to_detail=0.6,
            assistive_tech_readiness=0.6, assessment_performance=0.5),
        "assessment_score": 0.55, "training_completion": 0.5,
        "accommodation_compatibility": 0.82,
        "accommodations": ["accessible testing tools",
                           "screen-reader compatible dashboards",
                           "keyboard-navigable work environments"],
    },
    {
        "slug": "education_pathway_4_asd_software_testing",
        "title": "Autism Spectrum Disorder -> Junior Software Testing Assistant",
        "disability_label": "autism spectrum disorder",
        "disability_category": "communication",
        "target_role": "junior_software_testing_assistant",
        "language": "en",
        "competencies": _competency_vector(
            digital_literacy=0.6, test_case_execution=0.4, defect_reporting=0.35,
            attention_to_detail=0.7, problem_solving=0.5, workplace_behavior=0.4,
            assessment_performance=0.5),
        "assessment_score": 0.6, "training_completion": 0.5,
        "accommodation_compatibility": 0.78,
        "accommodations": ["written task instructions", "reduced sensory distraction",
                           "structured feedback", "clearly defined supervisor channels"],
    },
]


def render_demo(agent: EducationAgent, scenario: dict) -> str:
    out = agent.run(
        user_id=scenario["slug"],
        competencies=scenario["competencies"],
        target_role=scenario["target_role"],
        disability_category=scenario["disability_category"],
        language=scenario["language"],
        assessment_score=scenario["assessment_score"],
        training_completion=scenario["training_completion"],
        accommodation_compatibility=scenario["accommodation_compatibility"],
    )
    meta = out.metadata
    adaptations = DISABILITY_ADAPTATIONS.get(scenario["disability_category"], [])

    lines: list[str] = []
    lines.append(f"# Education-to-Employment Pathway — {scenario['title']}\n")
    lines.append("> Generated by the rule-based EducationAgent (FedAgent-Chain §5). "
                 "All learner data stays local; only protected model updates leave the node.\n")
    lines.append("## Learner & Target")
    lines.append(f"- **Disability profile:** {scenario['disability_label']}")
    lines.append(f"- **Target role:** `{scenario['target_role']}`")
    lines.append(f"- **Preferred language:** {scenario['language']}\n")

    lines.append("## Skill-Gap Analysis")
    lines.append(f"- **Skill-gap magnitude ‖G(i,r)‖₁:** {meta['skill_gap_l1']}")
    if meta["top_priority_gaps"]:
        lines.append("- **Top priority competency gaps:**")
        for g in meta["top_priority_gaps"]:
            lines.append(f"  - `{g['competency']}` (gap {g['gap']}, priority {g['priority']})")
    lines.append("")

    lines.append("## Recommended Accessible Learning Pathway")
    for i, r in enumerate(out.recommendations, 1):
        aligned = " (role-aligned)" if r["role_aligned"] else ""
        lines.append(f"{i}. **{r['title']}** [{r['resource_id']}] — score {r['score']}{aligned}")
    lines.append("")
    if adaptations:
        lines.append("### Disability-specific adaptations applied")
        lines.append(", ".join(adaptations) + "\n")

    lines.append("## Workplace-Readiness Evaluation")
    decision = ("ready for human-reviewed employment transition"
                if meta["ready_for_transition"]
                else "below threshold — another adaptive learning cycle recommended")
    lines.append(f"- **Readiness W(i,r):** {meta['readiness']} "
                 f"(threshold b_r = {meta['readiness_threshold']})")
    lines.append(f"- **Decision:** {decision}")
    lines.append(f"- **Governance review required:** {out.requires_human_review} "
                 f"(risk {round(out.risk_score, 3)})\n")

    lines.append("## Workplace Accommodation Plan (shared minimally with employer)")
    for a in scenario["accommodations"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append(f"_Explanation:_ {out.explanation}\n")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=str, default="experiments/results/demos/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(format="console")
    out_dir = ensure_dir(Path(args.output_dir))
    agent = EducationAgent()
    for scenario in SCENARIOS:
        content = render_demo(agent, scenario)
        path = out_dir / f"{scenario['slug']}.md"
        path.write_text(content, encoding="utf-8")
        logger.info("Education demo written", path=str(path))
    print(f"\n[OK] {len(SCENARIOS)} education-to-employment pathway demos written to {out_dir}")


if __name__ == "__main__":
    main()
