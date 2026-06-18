"""Education-to-employment catalog for FedAgent-Chain (paper Section 5).

Defines the static domain knowledge used by the education agent:

- ``COMPETENCIES`` — the 20 job-relevant competency dimensions that make up the
  learner competency vector ``s_i`` and the role requirement vector ``q_r``.
- ``JOB_ROLE_REQUIREMENTS`` — for each supported accessible job role: the required
  competency vector ``q_r`` (levels in [0, 1]), the per-competency importance
  vector ``I_{r,j}``, the minimum job-readiness threshold ``b_r``, and the work
  modes through which the role can be performed.
- ``LEARNING_RESOURCES`` — an accessible micro-learning library; each resource
  declares the competencies it covers, the roles it aligns with, the accessibility
  modes it supports, and its language.
- ``DISABILITY_ADAPTATIONS`` — disability-specific pathway adaptations used when
  rendering the illustrative learning pathways.

These are deliberately rule-based / curated (no LLMs), consistent with the
prototype's "rule-based and score-based agents" design.
"""

from __future__ import annotations

# ── Competency dimensions (m = 20) ─────────────────────────────────────────────
COMPETENCIES: list[str] = [
    "digital_literacy",  # 0
    "language_proficiency",  # 1
    "written_communication",  # 2
    "verbal_communication",  # 3
    "task_accuracy",  # 4
    "problem_solving",  # 5
    "attention_to_detail",  # 6
    "time_management",  # 7
    "data_entry",  # 8
    "spreadsheet_skills",  # 9
    "customer_service",  # 10
    "ticket_handling",  # 11
    "accessibility_testing",  # 12
    "screen_reader_use",  # 13
    "keyboard_navigation",  # 14
    "defect_reporting",  # 15
    "test_case_execution",  # 16
    "workplace_behavior",  # 17
    "assistive_tech_readiness",  # 18
    "assessment_performance",  # 19
]
N_COMPETENCIES = len(COMPETENCIES)


def _vec(**levels: float) -> list[float]:
    """Build a length-20 competency vector from named levels (default 0.0)."""
    v = [0.0] * N_COMPETENCIES
    for name, lvl in levels.items():
        v[COMPETENCIES.index(name)] = float(lvl)
    return v


# ── Supported accessible job roles ─────────────────────────────────────────────
# required = q_r (target competency levels); importance = I_{r,j}; b_r threshold.
JOB_ROLE_REQUIREMENTS: dict[str, dict] = {
    "remote_customer_support_officer": {
        "required": _vec(
            digital_literacy=0.7,
            language_proficiency=0.7,
            written_communication=0.8,
            customer_service=0.8,
            ticket_handling=0.7,
            workplace_behavior=0.6,
            time_management=0.6,
            assessment_performance=0.6,
        ),
        "importance": _vec(
            written_communication=1.0,
            customer_service=1.0,
            ticket_handling=0.9,
            language_proficiency=0.8,
            digital_literacy=0.6,
        ),
        "min_readiness": 0.62,
        "work_modes": ["remote", "hybrid"],
    },
    "data_entry_assistant": {
        "required": _vec(
            digital_literacy=0.6,
            data_entry=0.85,
            spreadsheet_skills=0.8,
            task_accuracy=0.85,
            attention_to_detail=0.8,
            time_management=0.6,
            assessment_performance=0.6,
        ),
        "importance": _vec(
            data_entry=1.0,
            task_accuracy=1.0,
            attention_to_detail=0.9,
            spreadsheet_skills=0.8,
            digital_literacy=0.5,
        ),
        "min_readiness": 0.60,
        "work_modes": ["remote", "hybrid", "onsite"],
    },
    "digital_accessibility_tester": {
        "required": _vec(
            digital_literacy=0.7,
            accessibility_testing=0.85,
            screen_reader_use=0.8,
            keyboard_navigation=0.8,
            defect_reporting=0.75,
            attention_to_detail=0.8,
            assistive_tech_readiness=0.7,
            assessment_performance=0.6,
        ),
        "importance": _vec(
            accessibility_testing=1.0,
            screen_reader_use=0.9,
            keyboard_navigation=0.9,
            defect_reporting=0.8,
            attention_to_detail=0.7,
        ),
        "min_readiness": 0.64,
        "work_modes": ["remote", "hybrid"],
    },
    "administrative_assistant": {
        "required": _vec(
            digital_literacy=0.7,
            written_communication=0.7,
            spreadsheet_skills=0.6,
            time_management=0.7,
            task_accuracy=0.7,
            workplace_behavior=0.6,
            assessment_performance=0.6,
        ),
        "importance": _vec(
            time_management=1.0, written_communication=0.9, task_accuracy=0.8, digital_literacy=0.7
        ),
        "min_readiness": 0.60,
        "work_modes": ["onsite", "hybrid"],
    },
    "e_commerce_support_worker": {
        "required": _vec(
            digital_literacy=0.7,
            customer_service=0.7,
            written_communication=0.7,
            ticket_handling=0.6,
            task_accuracy=0.7,
            assessment_performance=0.6,
        ),
        "importance": _vec(
            customer_service=1.0,
            written_communication=0.8,
            ticket_handling=0.7,
            digital_literacy=0.6,
        ),
        "min_readiness": 0.60,
        "work_modes": ["remote", "hybrid"],
    },
    "content_moderation_assistant": {
        "required": _vec(
            digital_literacy=0.7,
            attention_to_detail=0.8,
            problem_solving=0.6,
            workplace_behavior=0.7,
            task_accuracy=0.7,
            assessment_performance=0.6,
        ),
        "importance": _vec(
            attention_to_detail=1.0,
            problem_solving=0.8,
            workplace_behavior=0.8,
            digital_literacy=0.6,
        ),
        "min_readiness": 0.61,
        "work_modes": ["remote"],
    },
    "junior_software_testing_assistant": {
        "required": _vec(
            digital_literacy=0.7,
            test_case_execution=0.8,
            defect_reporting=0.75,
            attention_to_detail=0.8,
            problem_solving=0.65,
            workplace_behavior=0.6,
            assessment_performance=0.6,
        ),
        "importance": _vec(
            test_case_execution=1.0,
            defect_reporting=0.9,
            attention_to_detail=0.8,
            problem_solving=0.7,
            digital_literacy=0.6,
        ),
        "min_readiness": 0.63,
        "work_modes": ["remote", "hybrid"],
    },
}


# ── Accessible micro-learning resource library ─────────────────────────────────
# accessibility_modes — supported delivery adaptations (used for Acc(i,l)).
LEARNING_RESOURCES: list[dict] = [
    {
        "resource_id": "R01",
        "title": "Spreadsheet Fundamentals (adaptive)",
        "competencies": ["spreadsheet_skills", "data_entry", "digital_literacy"],
        "roles": ["data_entry_assistant", "administrative_assistant"],
        "accessibility_modes": {"simplified_text", "step_by_step", "screen_reader", "captioned"},
        "language": "multi",
    },
    {
        "resource_id": "R02",
        "title": "Accurate Data Entry & Error Checking",
        "competencies": ["data_entry", "task_accuracy", "attention_to_detail"],
        "roles": ["data_entry_assistant"],
        "accessibility_modes": {"simplified_text", "repeated_practice", "visual_prompts"},
        "language": "multi",
    },
    {
        "resource_id": "R03",
        "title": "Written Customer Communication",
        "competencies": ["written_communication", "customer_service", "ticket_handling"],
        "roles": ["remote_customer_support_officer", "e_commerce_support_worker"],
        "accessibility_modes": {"captioned", "text_based", "templates"},
        "language": "multi",
    },
    {
        "resource_id": "R04",
        "title": "Complaint Handling Role-Play (captioned)",
        "competencies": ["customer_service", "verbal_communication", "ticket_handling"],
        "roles": ["remote_customer_support_officer"],
        "accessibility_modes": {"captioned", "transcript", "text_based"},
        "language": "multi",
    },
    {
        "resource_id": "R05",
        "title": "Screen-Reader Testing Procedures",
        "competencies": ["screen_reader_use", "accessibility_testing", "assistive_tech_readiness"],
        "roles": ["digital_accessibility_tester"],
        "accessibility_modes": {"screen_reader", "audio", "keyboard_only"},
        "language": "multi",
    },
    {
        "resource_id": "R06",
        "title": "Keyboard-Only Navigation & Defect Reporting",
        "competencies": ["keyboard_navigation", "defect_reporting", "accessibility_testing"],
        "roles": ["digital_accessibility_tester", "junior_software_testing_assistant"],
        "accessibility_modes": {"keyboard_only", "templates", "audio"},
        "language": "multi",
    },
    {
        "resource_id": "R07",
        "title": "Structured Test-Case Execution",
        "competencies": ["test_case_execution", "attention_to_detail", "task_accuracy"],
        "roles": ["junior_software_testing_assistant"],
        "accessibility_modes": {"structured_tasks", "visual_schedule", "low_ambiguity"},
        "language": "multi",
    },
    {
        "resource_id": "R08",
        "title": "Defect Classification & Reporting",
        "competencies": ["defect_reporting", "problem_solving", "written_communication"],
        "roles": ["junior_software_testing_assistant", "digital_accessibility_tester"],
        "accessibility_modes": {"structured_tasks", "templates", "low_ambiguity"},
        "language": "multi",
    },
    {
        "resource_id": "R09",
        "title": "Workplace Behavior & Communication Practice",
        "competencies": ["workplace_behavior", "verbal_communication", "time_management"],
        "roles": [
            "junior_software_testing_assistant",
            "content_moderation_assistant",
            "administrative_assistant",
        ],
        "accessibility_modes": {"visual_schedule", "predictable_structure", "scenario_based"},
        "language": "multi",
    },
    {
        "resource_id": "R10",
        "title": "Digital Literacy Essentials",
        "competencies": ["digital_literacy", "time_management", "assessment_performance"],
        "roles": list(JOB_ROLE_REQUIREMENTS.keys()),
        "accessibility_modes": {"simplified_text", "audio", "captioned", "screen_reader"},
        "language": "multi",
    },
    {
        "resource_id": "R11",
        "title": "Attention & Detail Drills (content moderation)",
        "competencies": ["attention_to_detail", "problem_solving", "workplace_behavior"],
        "roles": ["content_moderation_assistant"],
        "accessibility_modes": {"low_distraction", "repeated_practice", "structured_tasks"},
        "language": "multi",
    },
    {
        "resource_id": "R12",
        "title": "Assistive Technology Readiness",
        "competencies": ["assistive_tech_readiness", "screen_reader_use", "keyboard_navigation"],
        "roles": ["digital_accessibility_tester", "remote_customer_support_officer"],
        "accessibility_modes": {"audio", "screen_reader", "keyboard_only"},
        "language": "multi",
    },
]


# ── Disability-specific pathway adaptations (used for illustrative demos) ───────
DISABILITY_ADAPTATIONS: dict[str, list[str]] = {
    "cognitive": [
        "short micro-lessons",
        "simplified text",
        "repeated examples",
        "visual prompts",
        "step-by-step instructions",
        "low-distraction interfaces",
        "frequent formative feedback",
    ],
    "hearing": [
        "captioned video material",
        "text-based communication exercises",
        "transcript-supported role-play",
        "written customer-service templates",
    ],
    "vision": [
        "screen-reader compatible content",
        "keyboard-only exercises",
        "audio instructions",
        "accessible assessment interfaces",
    ],
    "mobility": [
        "alternative input methods",
        "adjustable task duration",
        "voice-based interaction",
        "reduced fine-motor interaction requirements",
    ],
    "communication": [
        "predictable task structures",
        "explicit instructions",
        "visual schedules",
        "scenario-based workplace communication practice",
        "low-ambiguity assessment tasks",
    ],
}

# Map a disability category to the accessibility modes its learners benefit from.
DISABILITY_ACCESSIBILITY_MODES: dict[str, set] = {
    "cognitive": {
        "simplified_text",
        "step_by_step",
        "repeated_practice",
        "visual_prompts",
        "low_distraction",
    },
    "hearing": {"captioned", "text_based", "transcript", "templates"},
    "vision": {"screen_reader", "audio", "keyboard_only"},
    "mobility": {"keyboard_only", "audio", "structured_tasks"},
    "communication": {
        "structured_tasks",
        "visual_schedule",
        "predictable_structure",
        "low_ambiguity",
        "scenario_based",
    },
}
