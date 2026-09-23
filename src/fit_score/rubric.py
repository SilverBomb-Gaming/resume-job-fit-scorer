"""Rubric loading. The built-in rubric is used when --rubric is omitted."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fit_score.errors import RubricError

DEFAULT_DESCRIPTION = (
    "Extract concrete job-description requirements into categories, "
    "then compare each one to the resume."
)


class RubricCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def strip_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        cleaned = dict(value)
        for key in ("id", "label", "description"):
            if isinstance(cleaned.get(key), str):
                cleaned[key] = cleaned[key].strip()
        return cleaned


class RubricWeights(BaseModel):
    """Relative importance of must-have and nice-to-have coverage."""

    model_config = ConfigDict(extra="forbid")

    must: float = Field(gt=0)
    nice: float = Field(gt=0)


class StatusWeights(BaseModel):
    """Credit given to each coverage status before the result is scaled to 0–100."""

    model_config = ConfigDict(extra="forbid")

    matched: float = Field(ge=0)
    partial: float = Field(ge=0)
    missing: float = Field(ge=0)


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    categories: list[RubricCategory] = Field(min_length=1)
    weights: RubricWeights
    status_weights: StatusWeights

    @model_validator(mode="after")
    def unique_category_ids(self) -> Rubric:
        ids = [category.id.lower() for category in self.categories]
        if len(ids) != len(set(ids)):
            raise ValueError("Rubric category ids must be unique.")
        return self


def default_rubric() -> Rubric:
    """Sensible default: requirements, skills, years, and tools from the JD."""
    return Rubric(
        name="default",
        description=DEFAULT_DESCRIPTION,
        categories=[
            RubricCategory(
                id="requirements",
                label="Requirements",
                description=(
                    "Explicit responsibilities and qualifications the job "
                    "description states as requirements."
                ),
            ),
            RubricCategory(
                id="skills",
                label="Skills",
                description=(
                    "Domain skills, methods, and competencies named in the job description."
                ),
            ),
            RubricCategory(
                id="years",
                label="Years of experience",
                description=(
                    "Years of experience, seniority, and how long the candidate "
                    "should have practiced a skill."
                ),
            ),
            RubricCategory(
                id="tools",
                label="Tools",
                description="Named tools, languages, protocols, platforms, and systems.",
            ),
        ],
        weights=RubricWeights(must=0.75, nice=0.25),
        status_weights=StatusWeights(matched=1.0, partial=0.5, missing=0.0),
    )


def load_rubric(path: Path) -> Rubric:
    """Load a YAML or JSON rubric file."""
    if not path.exists() or not path.is_file():
        raise RubricError(f"Rubric not found: {path}")
    suffix = path.suffix.lower()
    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RubricError(f"Rubric {path} is not valid UTF-8 text.") from exc
    try:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(raw_text)
        elif suffix == ".json":
            data = json.loads(raw_text)
        else:
            raise RubricError("Rubric must be a .yaml, .yml, or .json file.")
    except yaml.YAMLError as exc:
        raise RubricError(f"Could not parse rubric YAML: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RubricError(f"Could not parse rubric JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RubricError("Rubric file must contain an object.")
    try:
        return Rubric.model_validate(data)
    except ValidationError as exc:
        raise RubricError(f"Invalid rubric: {exc}") from exc
