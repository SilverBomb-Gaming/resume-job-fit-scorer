"""CLI argument handling. The model client is patched; nothing listens on localhost."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fit_score.cli import app
from fit_score.errors import LLMError
from tests.support import ScriptedClient

runner = CliRunner()


def test_score_help_lists_inputs() -> None:
    result = runner.invoke(app, ["score", "--help"])
    assert result.exit_code == 0
    for flag in ("--resume", "--jd", "--resume-text", "--jd-text", "--json", "--rubric"):
        assert flag in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "fit-score 0.1.0" in result.stdout


def test_missing_both_inputs() -> None:
    result = runner.invoke(app, ["score"])
    assert result.exit_code == 1
    assert "resume" in result.stderr.lower()


def test_rejects_pdf(tmp_path: Path) -> None:
    resume = tmp_path / "resume.pdf"
    jd = tmp_path / "jd.md"
    resume.write_bytes(b"%PDF")
    jd.write_text("Python required\n", encoding="utf-8")
    result = runner.invoke(app, ["score", "--resume", str(resume), "--jd", str(jd)])
    assert result.exit_code == 1
    assert "PDF" in result.stderr


def test_rejects_both_resume_forms(tmp_path: Path) -> None:
    jd = tmp_path / "jd.md"
    resume = tmp_path / "resume.md"
    jd.write_text("Python required\n", encoding="utf-8")
    resume.write_text("Python\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["score", "--resume", str(resume), "--resume-text", "Python", "--jd", str(jd)],
    )
    assert result.exit_code == 1
    assert "--resume" in result.stderr


def test_text_flags_and_rubric_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = ScriptedClient()
    monkeypatch.setattr("fit_score.cli.build_client", lambda provider=None, model=None: client)
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(
        """
name: screen
description: Custom screen.
categories:
  - id: tools
    label: Tools
    description: Named tools.
weights:
  must: 0.75
  nice: 0.25
status_weights:
  matched: 1
  partial: 0.5
  missing: 0
""",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "score",
            "--resume-text",
            "RESUME-TOKEN-ZX9 knows Python.",
            "--jd-text",
            "JD-TOKEN-ZX9 requires Python.",
            "--rubric",
            str(rubric),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["rubric"] == "screen"
    assert "JD-TOKEN-ZX9" in client.calls[0]["user"]
    assert "RESUME-TOKEN-ZX9" not in client.calls[0]["user"]
    assert "RESUME-TOKEN-ZX9" in client.calls[1]["user"]
    assert "Extracting requirements" not in result.stdout


def test_stdin_supplies_the_job_description(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = ScriptedClient()
    monkeypatch.setattr("fit_score.cli.build_client", lambda provider=None, model=None: client)
    resume = tmp_path / "resume.md"
    resume.write_text("RESUME-TOKEN-ZX9\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["score", "--resume", str(resume), "--json"],
        input="STDIN-JD-TOKEN requires Python.\n",
    )
    assert result.exit_code == 0, result.stderr
    assert "STDIN-JD-TOKEN" in client.calls[0]["user"]
    assert "RESUME-TOKEN-ZX9" in client.calls[1]["user"]


def test_empty_stdin(tmp_path: Path) -> None:
    resume = tmp_path / "resume.md"
    resume.write_text("Python pytest\n", encoding="utf-8")
    result = runner.invoke(app, ["score", "--resume", str(resume)], input="   \n")
    assert result.exit_code == 1
    assert "empty" in result.stderr.lower()


def test_missing_rubric_file(tmp_path: Path) -> None:
    resume = tmp_path / "resume.md"
    jd = tmp_path / "jd.md"
    resume.write_text("Python\n", encoding="utf-8")
    jd.write_text("Python\n", encoding="utf-8")
    missing = tmp_path / "nope.yaml"
    result = runner.invoke(
        app,
        ["score", "--resume", str(resume), "--jd", str(jd), "--rubric", str(missing)],
    )
    assert result.exit_code == 2
    assert "Rubric not found" in result.stderr


def test_provider_failure_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def complete(self, *, system: str, user: str) -> str:
            raise LLMError("Could not reach Ollama at http://127.0.0.1:11434.")

        def close(self) -> None:
            return None

    monkeypatch.setattr("fit_score.cli.build_client", lambda provider=None, model=None: Boom())
    resume = tmp_path / "resume.md"
    jd = tmp_path / "jd.md"
    resume.write_text("Python\n", encoding="utf-8")
    jd.write_text("Python required\n", encoding="utf-8")
    result = runner.invoke(app, ["score", "--resume", str(resume), "--jd", str(jd)])
    assert result.exit_code == 2
    assert "Ollama" in result.stderr
