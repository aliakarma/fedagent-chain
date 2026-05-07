"""Data schema definitions and validators for FedAgent-Chain."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class NodeID(str, Enum):
    """Valid institutional node identifiers."""

    SAUDI_ARABIA = "saudi_arabia"
    UNITED_STATES = "united_states"
    CHINA = "china"
    EUROPE = "europe"


class DisabilityCategory(str, Enum):
    """Disability categories following WHO-ICF functional limitation classes."""

    MOBILITY = "mobility"
    VISION = "vision"
    HEARING = "hearing"
    COGNITIVE = "cognitive"
    COMMUNICATION = "communication"
    MENTAL_HEALTH = "mental_health"
    CHRONIC_HEALTH = "chronic_health"
    MULTIPLE = "multiple"


class WorkMode(str, Enum):
    """Preferred work modality."""

    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"


class EmploymentGoal(str, Enum):
    """Employment engagement type."""

    FULLTIME = "fulltime"
    PARTTIME = "parttime"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"


class EducationLevel(int, Enum):
    """Education level tiers (0=none, 4=postgraduate)."""

    NONE = 0
    SECONDARY = 1
    VOCATIONAL = 2
    UNDERGRADUATE = 3
    POSTGRADUATE = 4


class UserProfile(BaseModel):
    """Schema for a disability-employment user profile record.

    All fields must be present and valid. Records with consent_given=False
    are excluded from federated training by the preprocessing pipeline.

    Notes
    -----
    user_id is a synthetic UUID and contains no personally identifying information.
    skill_vector is a 50-dimensional binary vector derived from an occupational
    skills ontology (O*NET / ESCO crosswalk).
    accommodation_needs is a 20-dimensional binary vector of required accommodations.
    """

    user_id: str = Field(..., description="Synthetic UUID — no real PII")
    node_id: NodeID = Field(..., description="Regional institutional node")
    skill_vector: List[int] = Field(
        ..., min_length=50, max_length=50, description="50-dim binary skill vector"
    )
    education_level: EducationLevel
    disability_category: DisabilityCategory
    accommodation_needs: List[int] = Field(
        ..., min_length=20, max_length=20, description="20-dim binary accommodation flags"
    )
    language_primary: str = Field(..., min_length=2, max_length=5, description="ISO 639-1 code")
    language_secondary: Optional[str] = Field(
        None, min_length=2, max_length=5, description="ISO 639-1 code"
    )
    preferred_work_mode: WorkMode
    employment_goal: EmploymentGoal
    consent_given: bool = Field(
        ..., description="Must be True for record to be included in training"
    )

    @field_validator("skill_vector")
    @classmethod
    def validate_binary_skill_vector(cls, v: List[int]) -> List[int]:
        """Ensure skill vector contains only binary values."""
        if not all(x in (0, 1) for x in v):
            raise ValueError("skill_vector must contain only binary values (0 or 1)")
        return v

    @field_validator("accommodation_needs")
    @classmethod
    def validate_binary_accommodation(cls, v: List[int]) -> List[int]:
        """Ensure accommodation vector contains only binary values."""
        if not all(x in (0, 1) for x in v):
            raise ValueError("accommodation_needs must contain only binary values (0 or 1)")
        return v


class JobProfile(BaseModel):
    """Schema for a job opportunity profile record."""

    job_id: str = Field(..., description="Synthetic job UUID")
    node_id: NodeID = Field(..., description="Regional node where job is posted")
    required_skills: List[int] = Field(..., min_length=50, max_length=50)
    accessibility_score: float = Field(
        ..., ge=0.0, le=1.0, description="Job's accessibility/inclusivity rating"
    )
    work_mode: WorkMode
    language_required: str = Field(..., min_length=2, max_length=5)
    education_minimum: EducationLevel
    accommodation_provided: List[int] = Field(..., min_length=20, max_length=20)
    sector: str = Field(..., description="Industry/sector category")


class EmploymentOutcome(BaseModel):
    """Schema for a user-job suitability label record."""

    user_id: str
    job_id: str
    node_id: NodeID
    suitability_label: int = Field(..., ge=0, le=1, description="Binary match label")
    suitability_score: float = Field(
        ..., ge=0.0, le=1.0, description="Continuous suitability score [0, 1]"
    )


class BlockchainRecord(BaseModel):
    """Schema for a blockchain audit record (contains NO raw disability data)."""

    record_id: str
    node_id: str
    hash: str = Field(..., description="SHA-256 hash of protected model update")
    consent_ref: str = Field(..., description="Reference to consent record (no personal data)")
    policy_ref: str = Field(..., description="Applicable data governance policy reference")
    round_number: int = Field(..., ge=0)
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    status: str = Field(..., description="accepted | rejected | pending_review")

    @field_validator("hash")
    @classmethod
    def validate_sha256_format(cls, v: str) -> str:
        """Ensure hash is a valid 64-character hexadecimal string."""
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(f"Invalid SHA-256 hash format: {v}")
        return v
