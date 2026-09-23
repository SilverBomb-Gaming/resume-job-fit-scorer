"""Turn a resume and a job description into a scored report.

The model extracts requirements and judges coverage. This module computes the
0–100 score from the rubric so the number stays explainable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal, ROUND_HALF_UP
from typing import TypeVar

from pydantic import ValidationError

from fit_score.errors import ScoringError
from fit_score.llm import LLMClient
from fit_score.models import (
    Coverage,
    ExtractionResult,
    Judgment,
    JudgmentResult,
    LLMRewrite,
    Priority,
    RequirementAssessment,
    Rewrite,
    ScoreBreakdown,
    ScoreReport,
    Status,
)
from fit_score.prompts import SYSTEM_PROMPT, extraction_prompt, judgment_prompt
from fit_score.rubric import Rubric

T = TypeVar("T")

_REQUIREMENT_CAP = 16
_GENERIC_WHY_PREFIX = "Must-have coverage is"


def score_fit(
    resume_text: str,
    jd_text: str,
    rubric: Rubric,
    client: LLMClient,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> ScoreReport:
    """Extract JD requirements, compare them to the resume, and score the fit."""
    resume = resume_text.strip()
    job = jd_text.strip()
    if not resume or not job:
        raise ScoringError("Resume and job description must both be non-empty.")

    _progress(on_progress, "Extracting requirements from the job description...")
    grounded = _complete_json(
        client,
        system=SYSTEM_PROMPT,
        user=extraction_prompt(job, rubric),
        parse=lambda raw: _parse_extraction(raw, rubric),
        task="requirement extraction",
    )

    _progress(on_progress, "Comparing the resume to those requirements...")
    payload = [
        {
            "index": index,
            "text": item.text,
            "category": item.category,
            "priority": item.priority.value,
            "jd_evidence": item.jd_evidence,
        }
        for index, item in enumerate(grounded)
    ]
    judgment = _complete_json(
        client,
        system=SYSTEM_PROMPT,
        user=judgment_prompt(resume, payload),
        parse=_parse_judgment,
        task="resume comparison",
    )
    return assemble_report(rubric, grounded, judgment)


def assemble_report(
    rubric: Rubric,
    grounded: list[RequirementAssessment],
    judgment: JudgmentResult,
) -> ScoreReport:
    """Apply evidence rules and compute the score from judgments."""
    if not grounded:
        raise ScoringError("No JD-grounded requirements to score.")
    requirements = align_judgments(grounded, judgment.judgments)
    must = _coverage(requirements, Priority.must)
    nice = _coverage(requirements, Priority.nice)
    overall, breakdown = _score(requirements, rubric, must, nice)
    why = _why_bullets(must, nice, judgment.why)
    rewrites = _rewrites(judgment.rewrites)
    return ScoreReport(
        overall_score=overall,
        rubric=rubric.name,
        must_have=must,
        nice_to_have=nice,
        why=why,
        rewrites=rewrites,
        requirements=requirements,
        score_breakdown=breakdown,
    )


def align_judgments(
    grounded: list[RequirementAssessment],
    judgments: list[Judgment],
) -> list[RequirementAssessment]:
    """Pair judgments to extracted requirements.

    A matched or partial judgment with no resume quote becomes a gap. A missing
    judgment, or a missing index, is also a gap. The requirement text stays the
    one extracted from the job description.
    """
    normalized = _normalize_indexes(judgments, len(grounded))
    by_index: dict[int, Judgment] = {}
    for item in normalized:
        if item.index < 0 or item.index >= len(grounded):
            continue
        by_index.setdefault(item.index, item)

    aligned: list[RequirementAssessment] = []
    for index, requirement in enumerate(grounded):
        judgment = by_index.get(index)
        status = Status.missing
        evidence = ""
        if judgment is not None:
            status = judgment.status
            evidence = judgment.resume_evidence
            if status in {Status.matched, Status.partial} and not evidence:
                status = Status.missing
            if status == Status.missing:
                evidence = ""
        aligned.append(
            requirement.model_copy(update={"status": status, "resume_evidence": evidence})
        )
    return aligned


def ground_requirements(
    extracted: ExtractionResult,
    rubric: Rubric,
) -> list[RequirementAssessment]:
    """Keep unique requirements that quote the job description."""
    seen: set[str] = set()
    grounded: list[RequirementAssessment] = []
    for item in extracted.requirements:
        text = item.text.strip()
        evidence = item.jd_evidence.strip()
        if not text or not evidence:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        grounded.append(
            RequirementAssessment(
                id="pending",
                text=text,
                category=_normalize_category(item.category, rubric),
                priority=item.priority,
                status=Status.missing,
                jd_evidence=evidence,
                resume_evidence="",
            )
        )
    capped = _cap_requirements(grounded)
    return [
        item.model_copy(update={"id": f"req-{index}"})
        for index, item in enumerate(capped, start=1)
    ]


def _parse_extraction(raw: str, rubric: Rubric) -> list[RequirementAssessment]:
    data = extract_json_object(raw)
    parsed = ExtractionResult.model_validate(data)
    grounded = ground_requirements(parsed, rubric)
    if not grounded:
        raise ScoringError(
            "No requirements with job-description evidence were returned. "
            "Every requirement needs jd_evidence copied from the posting."
        )
    return grounded


def _parse_judgment(raw: str) -> JudgmentResult:
    data = extract_json_object(raw)
    parsed = JudgmentResult.model_validate(data)
    rewrites = [item for item in parsed.rewrites if item.suggested_bullet and item.grounded_in]
    why = [item for item in parsed.why if item]
    if len(rewrites) < 2:
        raise ScoringError(
            "Need at least two resume rewrites that name the resume fact they rephrase."
        )
    if not why:
        raise ScoringError("Need at least one why bullet.")
    return parsed.model_copy(update={"rewrites": rewrites[:4], "why": why[:5]})


def _complete_json(
    client: LLMClient,
    *,
    system: str,
    user: str,
    parse: Callable[[str], T],
    task: str,
) -> T:
    prompt = user
    last_error: Exception | None = None
    for _attempt in range(2):
        raw = client.complete(system=system, user=prompt)
        try:
            return parse(raw)
        except (ValidationError, ScoringError, ValueError, json.JSONDecodeError, TypeError) as exc:
            last_error = exc
            detail = str(exc).splitlines()[0][:400]
            prompt = (
                user
                + "\n\nThe previous response was invalid ("
                + detail
                + "). Return only one JSON object matching the schema. "
                + "Do not include markdown fences.\nPrevious response:\n"
                + raw[:1500]
            )
    raise ScoringError(
        f"The model returned an unusable {task} after a retry. Last error: {last_error}"
    ) from last_error


def extract_json_object(raw: str) -> dict[str, object]:
    """Parse a JSON object, tolerating a markdown fence or a short preamble."""
    text = raw.strip()
    if not text:
        raise ScoringError("The model returned an empty response.")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = json.loads(_first_json_object(text))
    if not isinstance(value, dict):
        raise ScoringError("The model returned JSON that is not an object.")
    return value


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ScoringError("The model response did not contain a JSON object.")
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ScoringError("The model response contained an incomplete JSON object.")


def _normalize_indexes(judgments: list[Judgment], count: int) -> list[Judgment]:
    """Shift a complete 1-based index set back to zero-based indexes."""
    if count == 0 or not judgments:
        return judgments
    indexes = [item.index for item in judgments]
    one_based = 0 not in indexes and min(indexes) >= 1 and max(indexes) == count
    if not one_based:
        return judgments
    return [item.model_copy(update={"index": item.index - 1}) for item in judgments]


def _normalize_category(category: str, rubric: Rubric) -> str:
    cleaned = category.strip().lower()
    by_id = {item.id.lower(): item.id for item in rubric.categories}
    if cleaned in by_id:
        return by_id[cleaned]
    by_label = {item.label.lower(): item.id for item in rubric.categories}
    if cleaned in by_label:
        return by_label[cleaned]
    return rubric.categories[0].id


def _cap_requirements(requirements: list[RequirementAssessment]) -> list[RequirementAssessment]:
    musts = [item for item in requirements if item.priority == Priority.must]
    nices = [item for item in requirements if item.priority == Priority.nice]
    if len(musts) >= _REQUIREMENT_CAP:
        return musts[:_REQUIREMENT_CAP]
    return musts + nices[: _REQUIREMENT_CAP - len(musts)]


def _coverage(requirements: list[RequirementAssessment], priority: Priority) -> Coverage:
    grouped = [item for item in requirements if item.priority == priority]
    matched = [item.text for item in grouped if item.status == Status.matched]
    partial = [item.text for item in grouped if item.status == Status.partial]
    missing = [item.text for item in grouped if item.status == Status.missing]
    return Coverage(
        matched=matched,
        partial=partial,
        missing=missing,
        matched_count=len(matched),
        partial_count=len(partial),
        missing_count=len(missing),
        total=len(grouped),
    )


def _score(
    requirements: list[RequirementAssessment],
    rubric: Rubric,
    must: Coverage,
    nice: Coverage,
) -> tuple[int, ScoreBreakdown]:
    status_weights = {
        Status.matched: Decimal(str(rubric.status_weights.matched)),
        Status.partial: Decimal(str(rubric.status_weights.partial)),
        Status.missing: Decimal(str(rubric.status_weights.missing)),
    }
    must_items = [item for item in requirements if item.priority == Priority.must]
    nice_items = [item for item in requirements if item.priority == Priority.nice]
    must_ratio = _ratio(must_items, status_weights)
    nice_ratio = _ratio(nice_items, status_weights)
    must_weight = Decimal(str(rubric.weights.must))
    nice_weight = Decimal(str(rubric.weights.nice))

    if must_ratio is None and nice_ratio is None:
        raw = Decimal("0")
        formula = "0 because no requirements were extracted"
    elif must_ratio is None:
        assert nice_ratio is not None
        raw = Decimal(100) * nice_ratio
        formula = f"100 * nice_ratio {_fmt(nice_ratio)} = {_fmt(raw)}"
    elif nice_ratio is None:
        raw = Decimal(100) * must_ratio
        formula = f"100 * must_ratio {_fmt(must_ratio)} = {_fmt(raw)}"
    else:
        raw = (
            Decimal(100)
            * (must_weight * must_ratio + nice_weight * nice_ratio)
            / (must_weight + nice_weight)
        )
        formula = (
            f"100 * ({_fmt(must_weight)} * {_fmt(must_ratio)} + "
            f"{_fmt(nice_weight)} * {_fmt(nice_ratio)}) / "
            f"({_fmt(must_weight)} + {_fmt(nice_weight)}) = {_fmt(raw)}"
        )

    rounded = int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    clamped = max(0, min(100, rounded))
    if clamped != rounded:
        formula = f"{formula}, clamped to {clamped}"
    breakdown = ScoreBreakdown(
        must_weight=float(must_weight),
        nice_weight=float(nice_weight),
        status_weights={status.value: float(weight) for status, weight in status_weights.items()},
        must_ratio=None if must_ratio is None else float(must_ratio.quantize(Decimal("0.0001"))),
        nice_ratio=None if nice_ratio is None else float(nice_ratio.quantize(Decimal("0.0001"))),
        formula=formula,
    )
    return clamped, breakdown


def _ratio(
    items: list[RequirementAssessment],
    status_weights: dict[Status, Decimal],
) -> Decimal | None:
    if not items:
        return None
    total = sum((status_weights[item.status] for item in items), Decimal("0"))
    return total / Decimal(len(items))


def _fmt(value: Decimal) -> str:
    quantized = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _why_bullets(must: Coverage, nice: Coverage, model_why: list[str]) -> list[str]:
    summary = (
        f"Must-have coverage is {must.matched_count} matched, "
        f"{must.partial_count} partial, and {must.missing_count} missing "
        f"out of {must.total}. "
        f"Nice-to-have coverage is {nice.matched_count} matched, "
        f"{nice.partial_count} partial, and {nice.missing_count} missing "
        f"out of {nice.total}."
    )
    extra = [item for item in model_why if not item.startswith(_GENERIC_WHY_PREFIX)]
    return [summary, *extra][:6]


def _rewrites(items: list[LLMRewrite]) -> list[Rewrite]:
    rewrites = [
        Rewrite(
            suggested_bullet=item.suggested_bullet,
            grounded_in=item.grounded_in,
            addresses=item.addresses,
        )
        for item in items
        if item.suggested_bullet and item.grounded_in
    ][:4]
    if len(rewrites) < 2:
        raise ScoringError(
            "Need at least two resume rewrites that name the resume fact they rephrase."
        )
    return rewrites


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
