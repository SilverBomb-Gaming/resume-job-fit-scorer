"""Shared fixtures for tests that mock the model."""

from __future__ import annotations

import json

EXTRACTION = {
    "requirements": [
        {
            "text": "Python test automation",
            "category": "skills",
            "priority": "must",
            "jd_evidence": "Python as a primary language",
        },
        {
            "text": "Modbus/TCP",
            "category": "tools",
            "priority": "must",
            "jd_evidence": "Modbus/TCP or serial instrument control",
        },
        {
            "text": "GitLab CI",
            "category": "tools",
            "priority": "must",
            "jd_evidence": "GitLab CI",
        },
        {
            "text": "24/7 NOC on-call",
            "category": "requirements",
            "priority": "must",
            "jd_evidence": "24/7 NOC on-call rotation",
        },
        {
            "text": "5+ years in test automation",
            "category": "years",
            "priority": "must",
            "jd_evidence": "5+ years in test automation",
        },
        {
            "text": "SQL for test results",
            "category": "skills",
            "priority": "nice",
            "jd_evidence": "SQL for querying test results",
        },
        {
            "text": "LabVIEW",
            "category": "tools",
            "priority": "nice",
            "jd_evidence": "LabVIEW is a plus",
        },
    ]
}

JUDGMENT = {
    "judgments": [
        {
            "index": 0,
            "status": "matched",
            "resume_evidence": "Built a Python pytest hardware-in-the-loop framework",
        },
        {
            "index": 1,
            "status": "matched",
            "resume_evidence": "Integrated Modbus/TCP and serial instrument drivers",
        },
        {
            "index": 2,
            "status": "partial",
            "resume_evidence": "Maintained Linux test stations",
        },
        {"index": 3, "status": "missing", "resume_evidence": ""},
        {
            "index": 4,
            "status": "matched",
            "resume_evidence": "6 years building manufacturing test software",
        },
        {
            "index": 5,
            "status": "matched",
            "resume_evidence": "Queried production test results with SQL",
        },
        {"index": 6, "status": "missing", "resume_evidence": ""},
    ],
    "why": [
        "Python, Modbus/TCP, and SQL are stated on the resume.",
        "24/7 NOC on-call is not mentioned.",
        "LabVIEW is not mentioned.",
    ],
    "rewrites": [
        {
            "suggested_bullet": (
                "Built a Python pytest hardware-in-the-loop framework that cut "
                "board-level functional test time by 30% on two SMT lines."
            ),
            "grounded_in": "cut board-level functional test time by 30%",
            "addresses": "Python test automation",
        },
        {
            "suggested_bullet": (
                "Integrated Modbus/TCP and serial instrument drivers for "
                "programmable power supplies, DAQs, and barcode scanners."
            ),
            "grounded_in": "Integrated Modbus/TCP and serial instrument drivers",
            "addresses": "Modbus/TCP",
        },
        {
            "suggested_bullet": (
                "Maintained Linux test stations and a GitLab CI pipeline that "
                "runs fixture smoke tests on every merge."
            ),
            "grounded_in": "GitLab CI pipeline that runs fixture smoke tests",
            "addresses": "GitLab CI",
        },
    ],
}

EXTRACTION_JSON = json.dumps(EXTRACTION)
JUDGMENT_JSON = json.dumps(JUDGMENT)

# must: matched, matched, partial, missing, matched → 3.5/5 = 0.7
# nice: matched, missing → 0.5
# 100 * (0.75 * 0.7 + 0.25 * 0.5) = 65
EXPECTED_SCORE = 65


class ScriptedClient:
    """Return canned extraction and judgment payloads."""

    def __init__(
        self,
        extraction: str = EXTRACTION_JSON,
        judgment: str = JUDGMENT_JSON,
    ) -> None:
        self.extraction = extraction
        self.judgment = judgment
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if "TASK: extract_requirements" in user:
            return self.extraction
        if "TASK: judge_resume" in user:
            return self.judgment
        raise AssertionError(user[:300])

    def close(self) -> None:
        return None


class RetryClient(ScriptedClient):
    """Fail the first extraction, then succeed so the retry path is exercised."""

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if "TASK: extract_requirements" in user:
            if "The previous response was invalid" not in user:
                return "this is not json"
            return self.extraction
        if "TASK: judge_resume" in user:
            return self.judgment
        raise AssertionError(user[:300])
