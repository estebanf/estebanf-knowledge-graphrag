"""Jev profiling (`PROFILING_BACKEND=jev`): question construction, probability
averaging, deterministic table/code detection, author/source LLM call and the
profile persistence helpers. No test touches the network."""

import json
from unittest.mock import MagicMock, patch

import pytest

from rag import jev, jev_profiling as jp
from rag.jev import JevError, JevResponse
from rag.profiling import DocumentProfile


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(jev, "_new_client", boom)
    monkeypatch.setattr("requests.post", boom)


def _choice_answer(options, winner, *, p=0.9):
    rest = [o for o in options if o != winner]
    probs = {winner: p}
    for option in rest:
        probs[option] = (1 - p) / len(rest)
    return {"type": "choice", "choice": winner, "probabilities": probs, "confidence": p}


def _all_answers(winners: dict[str, str], *, include_metadata=True, p=0.9):
    """Fake Jev answers: both order variants vote for `winners[field]`."""
    answers = {}
    specs = dict(jp.PROFILE_CHOICES)
    if include_metadata:
        specs.update(jp.METADATA_CHOICES)
    for name, (_, options) in specs.items():
        winner = winners.get(name, next(iter(options)))
        for key in jp._ab_keys(name):
            answers[key] = _choice_answer(list(options), winner, p=p)
    return answers


WINNERS = {
    "structure_type": "well-structured",
    "heading_consistency": "consistent",
    "content_density": "variable",
    "primary_content_type": "mixed",
    "avg_section_length": "long",
    "profile_domain": "policy",
    "kind": "guide",
    "metadata_domain": "technical",
}


# --- question building -------------------------------------------------------


def test_build_questions_asks_every_choice_twice_with_reversed_options():
    questions = jp.build_questions()

    assert len(questions) == 16  # 6 profile + kind + metadata_domain, twice each
    for name in list(jp.PROFILE_CHOICES) + list(jp.METADATA_CHOICES):
        a, b = questions[f"{name}__a"], questions[f"{name}__b"]
        assert a["type"] == b["type"] == "choice"
        assert a["instructions"] == b["instructions"]
        assert list(a["criteria"]) == list(reversed(list(b["criteria"])))
        assert a["criteria"] == b["criteria"]  # same descriptions, just reordered
        assert all(len(text) <= 255 for text in a["criteria"].values())


def test_profile_option_sets_match_document_profile_vocabulary():
    assert list(jp.PROFILE_CHOICES["structure_type"][1]) == [
        "well-structured", "loosely-structured", "unstructured",
    ]
    assert list(jp.PROFILE_CHOICES["heading_consistency"][1]) == ["consistent", "inconsistent", "none"]
    assert list(jp.PROFILE_CHOICES["content_density"][1]) == ["uniform", "variable"]
    assert list(jp.PROFILE_CHOICES["primary_content_type"][1]) == [
        "prose", "tabular", "mixed", "qa_pairs", "code", "transcript",
    ]
    assert list(jp.PROFILE_CHOICES["avg_section_length"][1]) == ["short", "medium", "long"]
    assert list(jp.PROFILE_CHOICES["profile_domain"][1]) == [
        "legal", "financial", "technical", "general", "medical", "policy",
    ]


def test_kind_and_metadata_domain_options():
    assert list(jp.METADATA_CHOICES["kind"][1]) == [
        "transcript", "paper", "article", "report", "infographic",
        "playbook", "guide", "policy", "specification", "other",
    ]
    # EXACTLY these 5 (feeds /api/sources/facets and frontend filters).
    assert set(jp.METADATA_CHOICES["metadata_domain"][1]) == {
        "legal", "financial", "technical", "medical", "general",
    }
    assert "policy" not in jp.METADATA_CHOICES["metadata_domain"][1]


def test_profile_domain_and_metadata_domain_are_separate_questions():
    questions = jp.build_questions()
    assert "profile_domain__a" in questions and "metadata_domain__a" in questions
    assert set(questions["profile_domain__a"]["criteria"]) != set(
        questions["metadata_domain__a"]["criteria"]
    )


def test_profile_only_questions_omit_metadata():
    questions = jp.build_questions(include_metadata=False)
    assert len(questions) == 12
    assert not any(key.startswith(("kind", "metadata_domain")) for key in questions)


# --- probability averaging ---------------------------------------------------


def test_average_choice_agreeing_orders():
    answers = {
        "f__a": _choice_answer(["x", "y", "z"], "y"),
        "f__b": _choice_answer(["z", "y", "x"], "y"),
    }
    assert jp.average_choice(answers, "f", ["x", "y", "z"])[0] == "y"


def test_average_choice_disagreeing_orders_uses_averaged_probabilities():
    # first-option bias: "a" order picks x (0.6 vs y 0.4); reversed order picks
    # y confidently (0.9 vs 0.1). Mean: x=0.35, y=0.65 -> y.
    answers = {
        "f__a": {"choice": "x", "probabilities": {"x": 0.6, "y": 0.4}},
        "f__b": {"choice": "y", "probabilities": {"x": 0.1, "y": 0.9}},
    }
    best, probability = jp.average_choice(answers, "f", ["x", "y"])
    assert best == "y"
    assert probability == pytest.approx(0.65)


def test_average_choice_disagreement_the_other_way():
    answers = {
        "f__a": {"choice": "x", "probabilities": {"x": 0.95, "y": 0.05}},
        "f__b": {"choice": "y", "probabilities": {"x": 0.45, "y": 0.55}},
    }
    best, probability = jp.average_choice(answers, "f", ["x", "y"])
    assert best == "x"
    assert probability == pytest.approx(0.7)


def test_average_choice_single_usable_answer_and_missing_probabilities():
    answers = {
        "f__a": {"choice": "y"},  # no probabilities -> full weight on chosen key
        "f__b": None,
    }
    assert jp.average_choice(answers, "f", ["x", "y"]) == ("y", 1.0)


def test_average_choice_ignores_unknown_option_keys():
    answers = {
        "f__a": {"choice": "bogus", "probabilities": {"bogus": 0.9, "x": 0.1}},
        "f__b": {"choice": "bogus", "probabilities": {"bogus": 0.8, "y": 0.2}},
    }
    # Only allowed options are considered: x=0.1, y=0.2 -> y.
    assert jp.average_choice(answers, "f", ["x", "y"])[0] == "y"


def test_average_choice_tie_resolves_to_canonical_first():
    answers = {
        "f__a": {"probabilities": {"x": 0.5, "y": 0.5}},
        "f__b": {"probabilities": {"x": 0.5, "y": 0.5}},
    }
    assert jp.average_choice(answers, "f", ["x", "y"])[0] == "x"


def test_average_choice_raises_without_usable_answers():
    with pytest.raises(jp.JevProfilingError):
        jp.average_choice({}, "f", ["x", "y"])
    with pytest.raises(jp.JevProfilingError):
        jp.average_choice({"f__a": {"choice": "nope", "probabilities": {"nope": 1.0}}}, "f", ["x"])


def test_metadata_domain_restricted_to_five_values():
    answers = _all_answers(WINNERS)
    # Jev "votes" for a value outside the allowed set; it must be ignored.
    for key in jp._ab_keys("metadata_domain"):
        answers[key] = {
            "choice": "policy",
            "probabilities": {"policy": 0.9, "legal": 0.02, "financial": 0.02,
                              "technical": 0.04, "medical": 0.01, "general": 0.01},
        }
    result = jp.classify_from_answers(answers, "# doc")
    assert result.metadata_domain == "technical"
    assert result.metadata_domain in jp.METADATA_DOMAINS


# --- deterministic features --------------------------------------------------


def test_detect_tables_and_code_blocks():
    table = "| a | b |\n|---|:---:|\n| 1 | 2 |\n"
    assert jp.detect_tables_and_code(table) == (True, False)
    assert jp.detect_tables_and_code("a | b\n--- | ---\n1 | 2\n") == (True, False)
    assert jp.detect_tables_and_code("```python\nprint(1)\n```\n") == (False, True)
    assert jp.detect_tables_and_code("~~~\nx\n~~~\n") == (False, True)
    assert jp.detect_tables_and_code("text\n\n---\n\nmore text\n") == (False, False)
    assert jp.detect_tables_and_code("") == (False, False)
    assert jp.detect_tables_and_code("# t\n\n" + table + "\n```\ncode\n```\n") == (True, True)


def test_horizontal_rules_and_bullets_are_not_tables():
    assert jp.detect_tables_and_code("* * *\n- - -\n-----\n| not a separator |\n") == (False, False)


def test_table_separator_inside_fenced_code_does_not_count():
    markdown = "```\n| a | b |\n|---|---|\n```\n"
    assert jp.detect_tables_and_code(markdown) == (False, True)


# --- classification ----------------------------------------------------------


def test_classify_from_answers_builds_profile_and_metadata():
    markdown = "| a | b |\n|---|---|\n\n```\nx\n```\n"
    result = jp.classify_from_answers(_all_answers(WINNERS), markdown, usage={"cost": 1e-6})

    assert result.profile == DocumentProfile(
        structure_type="well-structured",
        heading_consistency="consistent",
        content_density="variable",
        primary_content_type="mixed",
        avg_section_length="long",
        has_tables=True,
        has_code_blocks=True,
        domain="policy",
    )
    assert result.kind == "guide"
    assert result.metadata_domain == "technical"
    assert result.usage == {"cost": 1e-6}
    assert set(result.confidence) == set(WINNERS)


def test_classify_profile_only_has_no_metadata():
    result = jp.classify_from_answers(
        _all_answers(WINNERS, include_metadata=False), "text", include_metadata=False
    )
    assert result.kind is None and result.metadata_domain is None
    assert result.profile.domain == "policy"


def test_jev_classify_makes_one_call_on_the_excerpt(monkeypatch):
    calls = []

    def fake(state, questions, **kwargs):
        calls.append((state, questions))
        return JevResponse(answers=_all_answers(WINNERS), usage={"cost": 2e-6})

    monkeypatch.setattr(jev, "system_one_response", fake)
    markdown = "a" * 20000 + "\n```\ncode\n```\n"

    result = jp.jev_classify(markdown)

    assert len(calls) == 1
    state, questions = calls[0]
    assert len(state["document_excerpt"]) == jp.EXCERPT_CHARS
    assert set(questions) == set(jp.build_questions())
    # has_code_blocks is computed from the FULL markdown, not the excerpt
    assert result.profile.has_code_blocks is True
    assert result.usage == {"cost": 2e-6}


def test_jev_classify_empty_document_raises_without_calling_jev(monkeypatch):
    monkeypatch.setattr(jev, "system_one_response", MagicMock(side_effect=AssertionError))
    with pytest.raises(jp.JevProfilingError):
        jp.jev_classify("   \n")


def test_jev_classify_propagates_jev_errors(monkeypatch):
    monkeypatch.setattr(jev, "system_one_response", MagicMock(side_effect=JevError("down", 500)))
    with pytest.raises(JevError):
        jp.jev_classify("# doc")


# --- author/source LLM call --------------------------------------------------


def _llm_response(content):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def test_extract_author_source_asks_only_for_author_and_source(monkeypatch):
    from rag import metadata_extraction as me

    monkeypatch.setattr(me.settings, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(me.settings, "MODEL_METADATA_EXTRACTION", "meta/model")
    with patch(
        "rag.metadata_extraction.requests.post",
        return_value=_llm_response('```json\n{"author": "Jane", "source": "Gartner", "kind": "x"}\n```'),
    ) as post:
        result = me.extract_author_source("# Title\nBy Jane, Gartner")

    assert result == {"author": "Jane", "source": "Gartner"}
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "meta/model"
    prompt = payload["messages"][0]["content"]
    assert "author" in prompt and "source" in prompt
    assert "kind" not in prompt and "domain" not in prompt


def test_extract_author_source_returns_empty_on_error_or_no_key(monkeypatch):
    from rag import metadata_extraction as me

    monkeypatch.setattr(me.settings, "OPENROUTER_API_KEY", "test-key")
    with patch("rag.metadata_extraction.requests.post", side_effect=Exception("net")):
        assert me.extract_author_source("doc") == {}
    monkeypatch.setattr(me.settings, "OPENROUTER_API_KEY", "")
    assert me.extract_author_source("doc") == {}


def test_extract_profile_and_metadata_combines_jev_and_llm(monkeypatch):
    monkeypatch.setattr(
        jev, "system_one_response",
        lambda state, questions, **kw: JevResponse(answers=_all_answers(WINNERS)),
    )
    author_mock = MagicMock(return_value={"author": "Jane", "source": "Gartner"})
    monkeypatch.setattr(jp, "extract_author_source", author_mock)

    metadata, classification = jp.extract_profile_and_metadata("# doc")

    assert metadata == {"kind": "guide", "domain": "technical", "author": "Jane", "source": "Gartner"}
    assert classification.profile.domain == "policy"
    author_mock.assert_called_once_with("# doc")


def test_extract_profile_and_metadata_tolerates_author_source_failure(monkeypatch):
    monkeypatch.setattr(
        jev, "system_one_response",
        lambda state, questions, **kw: JevResponse(answers=_all_answers(WINNERS)),
    )
    monkeypatch.setattr(jp, "extract_author_source", lambda markdown: {})
    metadata, _ = jp.extract_profile_and_metadata("# doc")
    assert metadata == {"kind": "guide", "domain": "technical"}


def test_extract_profile_and_metadata_raises_when_jev_fails(monkeypatch):
    monkeypatch.setattr(jev, "system_one_response", MagicMock(side_effect=JevError("402", 402)))
    monkeypatch.setattr(jp, "extract_author_source", lambda markdown: {"author": "x"})
    with pytest.raises(JevError):
        jp.extract_profile_and_metadata("# doc")


# --- persistence helpers -----------------------------------------------------


def test_profile_dict_round_trip():
    profile = jp.classify_from_answers(_all_answers(WINNERS), "| a |\n|---|\n").profile
    data = jp.profile_to_dict(profile)
    json.dumps(data)  # JSON-serialisable for stage_log
    assert jp.profile_from_dict(data) == profile


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("domain"),
        lambda d: d.update(structure_type="weird"),
        lambda d: d.update(has_tables="yes"),
        lambda d: d.update(avg_section_length=None),
    ],
)
def test_profile_from_dict_rejects_malformed(mutate):
    data = jp.profile_to_dict(jp.classify_from_answers(_all_answers(WINNERS), "x").profile)
    mutate(data)
    assert jp.profile_from_dict(data) is None


def test_profile_from_dict_rejects_non_dict():
    assert jp.profile_from_dict(None) is None
    assert jp.profile_from_dict("nope") is None
    assert jp.profile_from_dict(MagicMock()) is None
