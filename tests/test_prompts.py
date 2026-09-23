"""The system prompt is the fabrication boundary."""

from fit_score.prompts import GUARDRAIL_PHRASES, SYSTEM_PROMPT, extraction_prompt, judgment_prompt
from fit_score.rubric import default_rubric


def test_system_prompt_contains_required_guardrails() -> None:
    for phrase in GUARDRAIL_PHRASES:
        assert phrase in SYSTEM_PROMPT


def test_guardrail_list_locks_the_required_phrases() -> None:
    required = {
        "Do not fabricate employers",
        "Do not fabricate skills",
        "Do not invent experience",
        "If the resume is silent",
        "mark it as a gap",
    }
    assert required <= set(GUARDRAIL_PHRASES)


def test_extraction_prompt_uses_rubric_categories_and_hides_the_resume() -> None:
    rubric = default_rubric()
    prompt = extraction_prompt("JD-ONLY-TOKEN must know Python.", rubric)
    assert "TASK: extract_requirements" in prompt
    for category in rubric.categories:
        assert category.id in prompt
    assert "JD-ONLY-TOKEN" in prompt
    assert "RESUME" not in prompt.split("JOB DESCRIPTION")[0]


def test_judgment_prompt_repeats_the_silence_rule() -> None:
    prompt = judgment_prompt(
        "Resume text",
        [
            {
                "index": 0,
                "text": "Python",
                "category": "skills",
                "priority": "must",
                "jd_evidence": "Python required",
            }
        ],
    )
    assert "TASK: judge_resume" in prompt
    assert "If the resume is silent" in prompt
    assert "mark it as a gap" in prompt
    assert "Do not invent experience" in prompt
    assert "Resume text" in prompt
