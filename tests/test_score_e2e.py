"""End-to-end score with a mocked model client. No Ollama process required."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fit_score.cli import app
from fit_score.models import ScoreReport
from fit_score.prompts import GUARDRAIL_PHRASES
from tests.support import EXPECTED_SCORE, ScriptedClient

runner = CliRunner()

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


def test_end_to_end_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = ScriptedClient()
    monkeypatch.setattr("fit_score.cli.build_client", lambda provider=None, model=None: client)
    resume = tmp_path / "resume.md"
    jd = tmp_path / "jd.md"
    resume.write_text("RESUME-TOKEN-ZX9\nBuilt a Python pytest framework.\n", encoding="utf-8")
    jd.write_text("JD-TOKEN-ZX9\nPython is required.\n", encoding="utf-8")

    json_result = runner.invoke(
        app,
        ["score", "--resume", str(resume), "--jd", str(jd), "--json"],
    )
    assert json_result.exit_code == 0, json_result.stderr
    payload = json.loads(json_result.stdout)
    report = ScoreReport.model_validate(payload)

    assert set(payload) == REPORT_KEYS
    assert report.overall_score == EXPECTED_SCORE
    assert report.rubric == "default"
    assert report.must_have.matched_count == 3
    assert "Python test automation" in report.must_have.matched
    assert report.must_have.partial == ["GitLab CI"]
    assert report.must_have.missing == ["24/7 NOC on-call"]
    assert report.nice_to_have.missing == ["LabVIEW"]
    assert report.why[0].startswith("Must-have coverage is 3 matched")
    assert 2 <= len(report.rewrites) <= 4
    assert all(item.grounded_in for item in report.rewrites)

    silent = next(item for item in report.requirements if item.text == "24/7 NOC on-call")
    assert silent.status.value == "missing"
    assert silent.resume_evidence == ""
    matched = next(item for item in report.requirements if item.text == "Python test automation")
    assert matched.resume_evidence

    assert len(client.calls) == 2
    for phrase in GUARDRAIL_PHRASES:
        assert phrase in client.calls[0]["system"]
        assert phrase in client.calls[1]["system"]
    assert "TASK: extract_requirements" in client.calls[0]["user"]
    assert "JD-TOKEN-ZX9" in client.calls[0]["user"]
    assert "RESUME-TOKEN-ZX9" not in client.calls[0]["user"]
    assert "TASK: judge_resume" in client.calls[1]["user"]
    assert "RESUME-TOKEN-ZX9" in client.calls[1]["user"]
    assert "Do not fabricate employers" in client.calls[0]["system"]

    text_result = runner.invoke(app, ["score", "--resume", str(resume), "--jd", str(jd)])
    assert text_result.exit_code == 0, text_result.stderr
    assert f"Fit score: {EXPECTED_SCORE} / 100" in text_result.stdout
    assert "Must-have coverage: 3 matched / 1 missing" in text_result.stdout
    assert "Python test automation" in text_result.stdout
    assert "LabVIEW" in text_result.stdout
    assert "Resume is silent." in text_result.stdout
    assert "Suggested resume bullet rewrites" in text_result.stdout
    assert "Why this score" in text_result.stdout
    assert "Extracting requirements" not in text_result.stdout
