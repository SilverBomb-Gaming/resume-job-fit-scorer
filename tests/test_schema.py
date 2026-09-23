"""JSON shape of a score report."""

import json

import pytest
from pydantic import ValidationError

from fit_score.models import (
    Coverage,
    ExtractionResult,
    Judgment,
    JudgmentResult,
    LLMRewrite,
    Priority,
    RequirementAssessment,
    ScoreReport,
    Status,
)
from fit_score.rubric import default_rubric
from fit_score.scoring import assemble_report, ground_requirements
from tests.support import EXPECTED_SCORE, EXTRACTION, JUDGMENT

REPORT_KEYS = {
    "overall_score",
    "rubric",
    "must_have",
    "nice_to_have",
    "why",
    "rewrites",
    "requirements",
    "score_breakdown",
}
COVERAGE_KEYS = {
    "matched",
    "partial",
    "missing",
    "matched_count",
    "partial_count",
    "missing_count",
    "total",
}
REQUIREMENT_KEYS = {
    "id",
    "text",
    "category",
    "priority",
    "status",
    "jd_evidence",
    "resume_evidence",
}
BREAKDOWN_KEYS = {
    "must_weight",
    "nice_weight",
    "status_weights",
    "must_ratio",
    "nice_ratio",
    "formula",
}


def _report() -> ScoreReport:
    grounded = ground_requirements(ExtractionResult.model_validate(EXTRACTION), default_rubric())
    judgment = JudgmentResult.model_validate(JUDGMENT)
    return assemble_report(default_rubric(), grounded, judgment)


def test_report_json_schema() -> None:
    report = _report()
    payload = json.loads(report.model_dump_json())
    assert set(payload) == REPORT_KEYS
    assert set(payload["must_have"]) == COVERAGE_KEYS
    assert set(payload["nice_to_have"]) == COVERAGE_KEYS
    assert set(payload["requirements"][0]) == REQUIREMENT_KEYS
    assert set(payload["score_breakdown"]) == BREAKDOWN_KEYS
    assert payload["overall_score"] == EXPECTED_SCORE
    assert isinstance(payload["overall_score"], int)
    assert payload["requirements"][0]["priority"] in {"must", "nice"}
    assert payload["requirements"][0]["status"] in {"matched", "partial", "missing"}
    assert payload["rewrites"][0]["grounded_in"]
    reloaded = ScoreReport.model_validate(payload)
    assert reloaded == report


def test_score_bounds_are_enforced() -> None:
    report = _report()
    payload = report.model_dump()
    payload["overall_score"] = 101
    with pytest.raises(ValidationError):
        ScoreReport.model_validate(payload)
    payload["overall_score"] = -1
    with pytest.raises(ValidationError):
        ScoreReport.model_validate(payload)


def test_coverage_counts_must_match_lists() -> None:
    with pytest.raises(ValidationError):
        Coverage(
            matched=["Python"],
            partial=[],
            missing=[],
            matched_count=0,
            partial_count=0,
            missing_count=0,
            total=1,
        )


def test_requirement_status_round_trip() -> None:
    item = RequirementAssessment(
        id="req-1",
        text="Python",
        category="skills",
        priority=Priority.must,
        status=Status.matched,
        jd_evidence="Python required",
        resume_evidence="Python pytest",
    )
    dumped = json.loads(item.model_dump_json())
    assert dumped["priority"] == "must"
    assert dumped["status"] == "matched"
    assert RequirementAssessment.model_validate(dumped) == item


def test_judgment_accepts_camel_case_evidence() -> None:
    judgment = Judgment.model_validate(
        {"index": "1", "status": "gap", "resumeEvidence": "should be cleared later"}
    )
    assert judgment.index == 1
    assert judgment.status == Status.missing
    rewrite = LLMRewrite.model_validate(
        {
            "suggestedBullet": "Built a Python framework.",
            "groundedIn": "Python framework",
        }
    )
    assert rewrite.suggested_bullet == "Built a Python framework."
    assert rewrite.grounded_in == "Python framework"
