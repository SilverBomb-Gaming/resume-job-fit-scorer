"""Command-line interface for the resume fit scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from fit_score import __version__
from fit_score.errors import LLMError, RubricError, ScoringError, UsageError
from fit_score.io import load_dotenv, resolve_inputs
from fit_score.llm import build_client
from fit_score.render import render_text
from fit_score.rubric import default_rubric, load_rubric
from fit_score.scoring import score_fit

app = typer.Typer(
    name="fit-score",
    help="Score how well a resume matches a job description.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"fit-score {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Score how well a resume matches a job description."""


@app.command()
def score(
    resume: Annotated[
        Optional[Path],
        typer.Option("--resume", help="Path to a .txt or .md resume."),
    ] = None,
    jd: Annotated[
        Optional[Path],
        typer.Option("--jd", help="Path to a .txt or .md job description."),
    ] = None,
    resume_text: Annotated[
        Optional[str],
        typer.Option("--resume-text", help="Resume text. Use this instead of --resume."),
    ] = None,
    jd_text: Annotated[
        Optional[str],
        typer.Option("--jd-text", help="Job description text. Use this instead of --jd."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
    rubric: Annotated[
        Optional[Path],
        typer.Option(
            "--rubric",
            help="YAML or JSON rubric. Uses the built-in default when omitted.",
        ),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", help="Model provider: ollama (default) or openai."),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", help="Model name. Overrides OLLAMA_MODEL or OPENAI_MODEL."),
    ] = None,
) -> None:
    """Score a resume against a job description.

    Omit either the resume or the job description and pipe that document on stdin.

    Examples:

      fit-score score --resume samples/resume.md --jd samples/jd-strong-fit.md

      fit-score score --resume samples/resume.md --jd samples/jd-weak-fit.md

      cat samples/jd-weak-fit.md | fit-score score --resume samples/resume.md --json
    """
    load_dotenv()
    try:
        resume_body, jd_body = resolve_inputs(
            resume=resume,
            jd=jd,
            resume_text=resume_text,
            jd_text=jd_text,
        )
        rubric_obj = load_rubric(rubric) if rubric is not None else default_rubric()
        client = build_client(provider, model)
        try:
            report = score_fit(
                resume_body,
                jd_body,
                rubric_obj,
                client,
                on_progress=lambda message: typer.echo(message, err=True),
            )
        finally:
            client.close()
    except UsageError as exc:
        _fail(str(exc), 1)
    except (RubricError, LLMError, ScoringError) as exc:
        _fail(str(exc), 2)

    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        typer.echo(render_text(report))


def _fail(message: str, code: int) -> None:
    typer.echo(f"fit-score: {message}", err=True)
    raise typer.Exit(code)
