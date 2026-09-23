"""Structured score report and the JSON the model is asked to return."""

from __future__ import annotations

from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class Priority(str, Enum):
    must = "must"
    nice = "nice"


class Status(str, Enum):
    matched = "matched"
    partial = "partial"
    missing = "missing"


_PRIORITY_ALIASES = {
    "must": Priority.must,
    "must-have": Priority.must,
    "must_have": Priority.must,
    "required": Priority.must,
    "requirement": Priority.must,
    "nice": Priority.nice,
    "nice-to-have": Priority.nice,
    "nice_to_have": Priority.nice,
    "preferred": Priority.nice,
    "optional": Priority.nice,
    "bonus": Priority.nice,
}

_STATUS_ALIASES = {
    "matched": Status.matched,
    "match": Status.matched,
    "met": Status.matched,
    "full": Status.matched,
    "yes": Status.matched,
    "partial": Status.partial,
    "partially matched": Status.partial,
    "partial match": Status.partial,
    "related": Status.partial,
    "missing": Status.missing,
    "gap": Status.missing,
    "unmatched": Status.missing,
    "no": Status.missing,
    "not found": Status.missing,
    "silent": Status.missing,
}


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


class Coverage(BaseModel):
    """Matched, partial, and missing requirement texts for one priority."""

    model_config = ConfigDict(extra="forbid")

    matched: list[str]
    partial: list[str]
    missing: list[str]
    matched_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    total: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_match_lists(self) -> Coverage:
        if self.matched_count != len(self.matched):
            raise ValueError("matched_count does not match matched.")
        if self.partial_count != len(self.partial):
            raise ValueError("partial_count does not match partial.")
        if self.missing_count != len(self.missing):
            raise ValueError("missing_count does not match missing.")
        expected = len(self.matched) + len(self.partial) + len(self.missing)
        if self.total != expected:
            raise ValueError("total does not match the coverage lists.")
        return self


class Rewrite(BaseModel):
    """A resume bullet that only restates a fact already in the resume."""

    model_config = ConfigDict(extra="forbid")

    suggested_bullet: str = Field(min_length=1)
    grounded_in: str = Field(min_length=1)
    addresses: str = ""


class RequirementAssessment(BaseModel):
    """One job requirement and how the resume covers it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    category: str
    priority: Priority
    status: Status
    jd_evidence: str
    resume_evidence: str = ""


class ScoreBreakdown(BaseModel):
    """The arithmetic behind overall_score. The model does not choose this number."""

    model_config = ConfigDict(extra="forbid")

    must_weight: float
    nice_weight: float
    status_weights: dict[str, float]
    must_ratio: float | None
    nice_ratio: float | None
    formula: str


class ScoreReport(BaseModel):
    """Machine-readable fit report printed with --json."""

    model_config = ConfigDict(extra="forbid")

    overall_score: int = Field(ge=0, le=100)
    rubric: str
    must_have: Coverage
    nice_to_have: Coverage
    why: list[str] = Field(min_length=1, max_length=6)
    rewrites: list[Rewrite] = Field(min_length=2, max_length=4)
    requirements: list[RequirementAssessment] = Field(min_length=1)
    score_breakdown: ScoreBreakdown


class ExtractedRequirement(BaseModel):
    """A requirement the model claims to have read in the job description."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    text: str = ""
    category: str = "requirements"
    priority: Priority
    jd_evidence: str = Field(
        default="",
        validation_alias=AliasChoices("jd_evidence", "jdEvidence"),
    )

    @field_validator("text", "category", "jd_evidence", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> str:
        return _clean(value)

    @field_validator("priority", mode="before")
    @classmethod
    def coerce_priority(cls, value: object) -> Priority:
        if isinstance(value, Priority):
            return value
        key = _clean(value).lower().replace(" ", "_")
        if key in _PRIORITY_ALIASES:
            return _PRIORITY_ALIASES[key]
        hyphenated = _clean(value).lower()
        if hyphenated in _PRIORITY_ALIASES:
            return _PRIORITY_ALIASES[hyphenated]
        raise ValueError(f"priority must be 'must' or 'nice', got {value!r}.")


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requirements: list[ExtractedRequirement] = Field(min_length=1)


class Judgment(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    index: int
    status: Status
    resume_evidence: str = Field(
        default="",
        validation_alias=AliasChoices("resume_evidence", "resumeEvidence"),
    )

    @field_validator("index", mode="before")
    @classmethod
    def coerce_index(cls, value: object) -> int:
        if isinstance(value, bool):
            raise ValueError("index must be an integer.")
        if isinstance(value, str):
            value = value.strip()
        return int(value)

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, value: object) -> Status:
        if isinstance(value, Status):
            return value
        key = _clean(value).lower()
        if key in _STATUS_ALIASES:
            return _STATUS_ALIASES[key]
        raise ValueError("status must be 'matched', 'partial', or 'missing'.")

    @field_validator("resume_evidence", mode="before")
    @classmethod
    def strip_evidence(cls, value: object) -> str:
        return _clean(value)


class LLMRewrite(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    suggested_bullet: str = Field(
        default="",
        validation_alias=AliasChoices("suggested_bullet", "suggestedBullet"),
    )
    grounded_in: str = Field(
        default="",
        validation_alias=AliasChoices("grounded_in", "groundedIn"),
    )
    addresses: str = ""

    @field_validator("suggested_bullet", "grounded_in", "addresses", mode="before")
    @classmethod
    def strip_fields(cls, value: object) -> str:
        return _clean(value)


class JudgmentResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    judgments: list[Judgment] = Field(min_length=1)
    why: list[str] = Field(default_factory=list)
    rewrites: list[LLMRewrite] = Field(default_factory=list)

    @field_validator("why", mode="before")
    @classmethod
    def strip_why(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("why must be a list of strings.")
        return [_clean(item) for item in value if _clean(item)]
