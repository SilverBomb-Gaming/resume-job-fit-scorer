"""Plain-text report. Machine-readable output is ScoreReport JSON from the CLI."""

from __future__ import annotations

from fit_score.models import Priority, RequirementAssessment, ScoreReport, Status


def render_text(report: ScoreReport) -> str:
    """Format a score report for a terminal."""
    lines = [
        f"Fit score: {report.overall_score} / 100",
        f"Rubric: {report.rubric}",
        f"Score math: {report.score_breakdown.formula}",
        "",
        _coverage_block("Must-have coverage", report, Priority.must),
        "",
        _coverage_block("Nice-to-have coverage", report, Priority.nice),
        "",
        "Why this score",
    ]
    for bullet in report.why:
        lines.append(f"  - {bullet}")
    lines.extend(
        [
            "",
            "Suggested resume bullet rewrites",
            "  These only rephrase facts already in the resume. They do not add employers, tools, or achievements.",
        ]
    )
    for index, rewrite in enumerate(report.rewrites, start=1):
        lines.append(f"  {index}. {rewrite.suggested_bullet}")
        lines.append(f"     Grounded in: {rewrite.grounded_in}")
        if rewrite.addresses:
            lines.append(f"     Addresses: {rewrite.addresses}")
    return "\n".join(lines)


def _coverage_block(title: str, report: ScoreReport, priority: Priority) -> str:
    grouped = [item for item in report.requirements if item.priority == priority]
    matched = [item for item in grouped if item.status == Status.matched]
    partial = [item for item in grouped if item.status == Status.partial]
    missing = [item for item in grouped if item.status == Status.missing]
    lines = [
        f"{title}: {len(matched)} matched / {len(missing)} missing",
        *_status_section("Matched", matched, include_resume=True),
        *_status_section("Partial", partial, include_resume=True),
        *_status_section("Missing", missing, include_resume=False),
    ]
    return "\n".join(lines)


def _status_section(
    label: str,
    items: list[RequirementAssessment],
    *,
    include_resume: bool,
) -> list[str]:
    lines = [f"  {label} ({len(items)})"]
    if not items:
        lines.append("    (none)")
        return lines
    for item in items:
        lines.append(f"    - [{item.category}] {item.text}")
        if include_resume:
            lines.append(f"      Resume: {item.resume_evidence}")
        else:
            lines.append("      Resume is silent.")
    return lines
