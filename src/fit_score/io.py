"""Document loading and a tiny .env reader. PDF is intentionally unsupported."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fit_score.errors import UsageError

ALLOWED_SUFFIXES = {".txt", ".md", ".markdown"}
MAX_CHARS = 80_000


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=value pairs into the environment without overriding existing vars."""
    env_path = path if path is not None else Path.cwd() / ".env"
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise UsageError(f"{env_path} is not valid UTF-8.") from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def read_document(path: Path) -> str:
    """Read a UTF-8 .txt or .md file."""
    if path.is_dir():
        raise UsageError(f"{path} is a directory. Pass a .txt or .md file.")
    if not path.exists() or not path.is_file():
        raise UsageError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raise UsageError(
            f"{path.name} is a PDF. PDF input is out of scope; export the document to .txt or .md."
        )
    if suffix not in ALLOWED_SUFFIXES:
        shown = suffix or "(none)"
        raise UsageError(f"Unsupported file type '{shown}' for {path.name}. Use .txt or .md.")
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UsageError(f"{path} is not valid UTF-8 text.") from exc


def resolve_inputs(
    *,
    resume: Path | None,
    jd: Path | None,
    resume_text: str | None,
    jd_text: str | None,
    stdin: object | None = None,
) -> tuple[str, str]:
    """Resolve resume and job-description text from paths, flags, or stdin.

    Stdin supplies whichever one of the two documents was omitted. It is not
    used when both documents were passed explicitly.
    """
    if resume is not None and resume_text is not None:
        raise UsageError("Pass only one of --resume and --resume-text.")
    if jd is not None and jd_text is not None:
        raise UsageError("Pass only one of --jd and --jd-text.")

    resume_body = resume_text if resume_text is not None else None
    jd_body = jd_text if jd_text is not None else None
    if resume is not None:
        resume_body = read_document(resume)
    if jd is not None:
        jd_body = read_document(jd)

    missing = [name for name, body in (("resume", resume_body), ("job description", jd_body)) if body is None]
    if len(missing) == 2:
        raise UsageError(
            "Provide a resume and a job description via --resume/--jd, "
            "--resume-text/--jd-text, or stdin for one of them."
        )
    if len(missing) == 1:
        stream = stdin if stdin is not None else sys.stdin
        if stream.isatty():
            raise UsageError(
                f"Missing {missing[0]}. Pass a path or text, or pipe that document on stdin."
            )
        piped = stream.read()
        if not piped or not piped.strip():
            raise UsageError(f"Stdin was empty; cannot read the {missing[0]}.")
        if resume_body is None:
            resume_body = piped
        else:
            jd_body = piped

    assert resume_body is not None and jd_body is not None
    resume_body = resume_body.strip()
    jd_body = jd_body.strip()
    if not resume_body or not jd_body:
        raise UsageError("Resume and job description must both be non-empty.")
    if len(resume_body) > MAX_CHARS:
        raise UsageError(
            f"Resume is too long ({len(resume_body)} characters). The limit is {MAX_CHARS}."
        )
    if len(jd_body) > MAX_CHARS:
        raise UsageError(
            f"Job description is too long ({len(jd_body)} characters). The limit is {MAX_CHARS}."
        )
    return resume_body, jd_body
