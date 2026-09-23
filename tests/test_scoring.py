"""Score arithmetic and evidence rules. No model calls except scripted fakes."""

import pytest

from fit_score.errors import ScoringError
from fit_score.models import (
    Judgment,
    JudgmentResult,
    LLMRewrite,
    Priority,
    RequirementAssessment,
    Status,
)
from fit_score.rubric import StatusWeights, default_rubric
from fit_score.scoring import align_judgments, assemble_report, extract_json_object, score_fit
from tests.support import EXTRACTION_JSON, EXPECTED_SCORE, JUDGMENT_JSON, RetryClient, ScriptedClient


def _requirement(text: str, *, priority: Priority = Priority.must) -> RequirementAssessment:
    return RequirementAssessment(
        id="req-1",
        text=text,
        category="skills",
        priority=priority,
        status=Status.missing,
        jd_evidence=f"JD says {text}",
        resume_evidence="",
    )


def _judgment(index: int, status: Status, evidence: str = "") -> JudgmentResult:
    return JudgmentResult(
        judgments=[Judgment(index=index, status=status, resume_evidence=evidence)],
        why=["Quoted from the documents."],
        rewrites=_two_rewrites(),
    )


def _two_rewrites() -> list[LLMRewrite]:
    return [
        LLMRewrite(
            suggested_bullet="Restate the Python fixture work already on the resume.",
            grounded_in="Python pytest hardware-in-the-loop framework",
            addresses="Python",
        ),
        LLMRewrite(
            suggested_bullet="Restate the Modbus driver work already on the resume.",
            grounded_in="Modbus/TCP and serial instrument drivers",
            addresses="Modbus",
        ),
    ]


def test_partial_credit_and_weights_produce_expected_score() -> None:
    client = ScriptedClient()
    report = score_fit("resume body", "job body", default_rubric(), client)
    assert report.overall_score == EXPECTED_SCORE
    assert report.must_have.matched_count == 3
    assert report.must_have.partial == ["GitLab CI"]
    assert report.must_have.missing == ["24/7 NOC on-call"]
    assert report.nice_to_have.matched == ["SQL for test results"]
    assert report.nice_to_have.missing == ["LabVIEW"]
    assert report.why[0].startswith("Must-have coverage is 3 matched, 1 partial, and 1 missing")
    assert len(report.rewrites) == 3
    assert len(client.calls) == 2


def test_equal_weights_change_the_score() -> None:
    rubric = default_rubric().model_copy(
        update={"weights": default_rubric().weights.model_copy(update={"must": 1, "nice": 1})}
    )
    client = ScriptedClient()
    report = score_fit("resume body", "job body", rubric, client)
    # 100 * (1 * 0.7 + 1 * 0.5) / 2 = 60
    assert report.overall_score == 60


def test_must_only_partial_scores_50() -> None:
    report = assemble_report(
        default_rubric(),
        [_requirement("Linux test stations")],
        _judgment(0, Status.partial, "Maintained Linux test stations"),
    )
    assert report.overall_score == 50
    assert report.score_breakdown.nice_ratio is None


def test_all_missing_must_scores_0() -> None:
    report = assemble_report(
        default_rubric(),
        [_requirement("SwiftUI")],
        _judgment(0, Status.missing, ""),
    )
    assert report.overall_score == 0
    assert report.must_have.missing == ["SwiftUI"]


def test_matched_without_resume_evidence_becomes_a_gap() -> None:
    requirements = [_requirement("SwiftUI")]
    aligned = align_judgments(
        requirements,
        [Judgment(index=0, status=Status.matched, resume_evidence="   ")],
    )
    assert aligned[0].status == Status.missing
    assert aligned[0].resume_evidence == ""
    report = assemble_report(
        default_rubric(),
        requirements,
        _judgment(0, Status.matched, "  "),
    )
    assert report.overall_score == 0


def test_missing_status_drops_a_resume_quote() -> None:
    aligned = align_judgments(
        [_requirement("SwiftUI")],
        [Judgment(index=0, status=Status.missing, resume_evidence="Invented SwiftUI job")],
    )
    assert aligned[0].status == Status.missing
    assert aligned[0].resume_evidence == ""


def test_omitted_index_is_a_gap() -> None:
    requirements = [
        _requirement("Python"),
        _requirement("SwiftUI").model_copy(update={"id": "req-2", "text": "SwiftUI"}),
    ]
    report = assemble_report(
        default_rubric(),
        requirements,
        _judgment(0, Status.matched, "Python pytest"),
    )
    assert report.requirements[1].status == Status.missing
    assert report.overall_score == 50


def test_one_based_indexes_are_shifted_when_the_set_is_complete() -> None:
    requirements = [
        _requirement("Python"),
        _requirement("Modbus").model_copy(update={"id": "req-2", "text": "Modbus"}),
    ]
    report = assemble_report(
        default_rubric(),
        requirements,
        JudgmentResult(
            judgments=[
                Judgment(index=1, status=Status.matched, resume_evidence="Python pytest"),
                Judgment(index=2, status=Status.matched, resume_evidence="Modbus/TCP"),
            ],
            why=["Both tools are on the resume."],
            rewrites=_two_rewrites(),
        ),
    )
    assert [item.status for item in report.requirements] == [Status.matched, Status.matched]
    assert report.overall_score == 100


def test_status_weight_above_one_clamps_at_100() -> None:
    rubric = default_rubric().model_copy(
        update={"status_weights": StatusWeights(matched=5, partial=0.5, missing=0)}
    )
    report = assemble_report(
        rubric,
        [_requirement("Python")],
        _judgment(0, Status.matched, "Python pytest"),
    )
    assert report.overall_score == 100
    assert "clamped to 100" in report.score_breakdown.formula


def test_fenced_and_preamble_json() -> None:
    fenced = ScriptedClient(
        extraction="```json\n" + EXTRACTION_JSON + "\n```",
        judgment="Here you go:\n" + JUDGMENT_JSON + "\nDone.",
    )
    report = score_fit("resume body", "job body", default_rubric(), fenced)
    assert report.overall_score == EXPECTED_SCORE


def test_invalid_extraction_is_retried() -> None:
    client = RetryClient()
    report = score_fit("resume body", "job body", default_rubric(), client)
    assert report.overall_score == EXPECTED_SCORE
    assert len(client.calls) == 3
    assert "The previous response was invalid" in client.calls[1]["user"]


def test_extract_json_object_finds_object_and_rejects_prose() -> None:
    assert extract_json_object('```json\n{"ok": true}\n```') == {"ok": True}
    assert extract_json_object('Note {"note": "brace } inside", "ok": true} tail')["ok"] is True
    with pytest.raises(ScoringError):
        extract_json_object("no json here")
