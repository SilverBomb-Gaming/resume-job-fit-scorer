"""Document loading and .env parsing."""

import os
from pathlib import Path

import pytest

from fit_score.errors import UsageError
from fit_score.io import load_dotenv, read_document, resolve_inputs


def test_read_markdown_and_text(tmp_path: Path) -> None:
    resume = tmp_path / "resume.md"
    resume.write_text("Python\n", encoding="utf-8")
    notes = tmp_path / "jd.txt"
    notes.write_text("pytest\n", encoding="utf-8")
    assert read_document(resume) == "Python\n"
    assert "pytest" in read_document(notes)


def test_pdf_is_out_of_scope(tmp_path: Path) -> None:
    path = tmp_path / "resume.pdf"
    path.write_bytes(b"%PDF-1.4")
    with pytest.raises(UsageError, match="PDF"):
        read_document(path)


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "resume.docx"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(UsageError, match="Unsupported"):
        read_document(path)


def test_dotenv_does_not_override_existing_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
# comment
export OLLAMA_MODEL=from-file
OPENAI_API_KEY="secret"
ALREADY=from-file
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALREADY", "from-env")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        load_dotenv(env_file)
        assert os.environ["OLLAMA_MODEL"] == "from-file"
        assert os.environ["OPENAI_API_KEY"] == "secret"
        assert os.environ["ALREADY"] == "from-env"
    finally:
        os.environ.pop("OLLAMA_MODEL", None)
        os.environ.pop("OPENAI_API_KEY", None)


def test_resolve_rejects_empty_pair() -> None:
    with pytest.raises(UsageError, match="non-empty"):
        resolve_inputs(resume=None, jd=None, resume_text="   ", jd_text="a real job")
