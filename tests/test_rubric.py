"""Built-in rubric and YAML/JSON loading."""

import json
from pathlib import Path

import pytest

from fit_score.errors import RubricError
from fit_score.rubric import default_rubric, load_rubric

ROOT = Path(__file__).resolve().parents[1]


def test_default_rubric_covers_requirements_skills_years_and_tools() -> None:
    rubric = default_rubric()
    assert rubric.name == "default"
    assert [category.id for category in rubric.categories] == [
        "requirements",
        "skills",
        "years",
        "tools",
    ]
    assert rubric.weights.must == 0.75
    assert rubric.weights.nice == 0.25
    assert rubric.status_weights.matched == 1.0
    assert rubric.status_weights.partial == 0.5
    assert rubric.status_weights.missing == 0.0


def test_sample_rubric_matches_the_builtin() -> None:
    loaded = load_rubric(ROOT / "samples" / "rubric.yaml")
    assert loaded == default_rubric()


def test_load_json_rubric(tmp_path: Path) -> None:
    payload = default_rubric().model_dump()
    payload["name"] = "screen"
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_rubric(path)
    assert loaded.name == "screen"
    assert [category.id for category in loaded.categories] == [
        "requirements",
        "skills",
        "years",
        "tools",
    ]


def test_load_yaml_rubric(tmp_path: Path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(
        """
name: lab
description: A smaller rubric.
categories:
  - id: tools
    label: Tools
    description: Named tools only.
weights:
  must: 1
  nice: 1
status_weights:
  matched: 1
  partial: 0
  missing: 0
""",
        encoding="utf-8",
    )
    loaded = load_rubric(path)
    assert loaded.name == "lab"
    assert loaded.categories[0].id == "tools"
    assert loaded.status_weights.partial == 0


def test_missing_rubric_raises() -> None:
    with pytest.raises(RubricError, match="not found"):
        load_rubric(Path("/tmp/does-not-exist-fit-score-rubric.yaml"))


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "rubric.txt"
    path.write_text("name: default\n", encoding="utf-8")
    with pytest.raises(RubricError, match="yaml"):
        load_rubric(path)


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "rubric.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(RubricError, match="JSON"):
        load_rubric(path)


def test_duplicate_category_ids_raise(tmp_path: Path) -> None:
    path = tmp_path / "rubric.json"
    path.write_text(
        json.dumps(
            {
                "name": "dup",
                "categories": [
                    {"id": "tools", "label": "Tools", "description": "One."},
                    {"id": "tools", "label": "Again", "description": "Two."},
                ],
                "weights": {"must": 1, "nice": 1},
                "status_weights": {"matched": 1, "partial": 0.5, "missing": 0},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RubricError, match="unique"):
        load_rubric(path)
