import math
from unittest.mock import patch, MagicMock
from rag.chunking import ChunkData
from rag.chunk_validation import validate_chunks

_BASE_CONFIG = {"chunk_size_tokens": 400, "chunk_overlap_tokens": 80}


def _make_chunk(content: str, index: int = 0) -> ChunkData:
    return ChunkData(
        content=content,
        token_count=len(content.split()),
        chunk_index=index,
        parent_chunk_id=None,
        chunking_strategy="recursive",
        chunking_config=_BASE_CONFIG,
    )


def _mock_pass_response():
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"choices": [{"message": {"content": '{"pass": true}'}}]}
    return mock


def _mock_fail_response():
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"choices": [{"message": {"content": '{"pass": false, "reason": "incomplete"}'}}]}
    return mock


def test_validate_chunks_no_api_key_always_passes(monkeypatch):
    monkeypatch.setattr("rag.chunk_validation.settings.OPENROUTER_API_KEY", "")
    chunks = [_make_chunk("Some content.", i) for i in range(5)]
    assert validate_chunks(chunks, domain="general") is True


def test_validate_chunks_all_pass(monkeypatch):
    monkeypatch.setattr("rag.chunk_validation.settings.OPENROUTER_API_KEY", "test-key")
    chunks = [_make_chunk("Some content.", i) for i in range(20)]
    with patch("rag.chunk_validation.requests.post", return_value=_mock_pass_response()):
        result = validate_chunks(chunks, domain="general")
    assert result is True


def test_validate_chunks_too_many_failures(monkeypatch):
    monkeypatch.setattr("rag.chunk_validation.settings.OPENROUTER_API_KEY", "test-key")
    chunks = [_make_chunk("Bad content.", i) for i in range(20)]
    with patch("rag.chunk_validation.requests.post", return_value=_mock_fail_response()):
        result = validate_chunks(chunks, domain="general")
    assert result is False


def test_validate_chunks_high_stakes_domain_samples_more(monkeypatch):
    monkeypatch.setattr("rag.chunk_validation.settings.OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("rag.chunk_validation.settings.CHUNK_VALIDATION_SAMPLE_RATE", 0.10)
    monkeypatch.setattr("rag.chunk_validation.settings.CHUNK_VALIDATION_SAMPLE_RATE_HIGH_STAKES", 0.25)
    chunks = [_make_chunk(f"Content {i}.", i) for i in range(100)]
    call_count = 0

    def counting_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _mock_pass_response()

    with patch("rag.chunk_validation.requests.post", side_effect=counting_post):
        validate_chunks(chunks, domain="legal")

    assert call_count == 25  # 25% of 100


# --- Jev backend (CHUNK_VALIDATION_BACKEND=jev) -----------------------------

import pytest

from rag import chunk_validation as cv


def _answers(complete=0.9, coherent=0.9, substantive=0.9):
    return {
        "complete": {"type": "noul", "noul": complete},
        "coherent": {"type": "noul", "noul": coherent},
        "substantive": {"type": "noul", "noul": substantive},
    }


@pytest.fixture
def jev_settings(monkeypatch):
    monkeypatch.setattr(cv.settings, "CHUNK_VALIDATION_JEV_THRESHOLD", 0.3)
    monkeypatch.setattr(cv.settings, "CHUNK_VALIDATION_MAX_SKIP_FRACTION", 0.5)
    monkeypatch.setattr(cv.settings, "JEV_CONCURRENCY", 5)


def _rows(n):
    return [(f"c{i}", f"content {i}") for i in range(n)]


def test_quality_questions_are_three_nouls_with_criteria():
    questions = cv.build_quality_questions()
    assert list(questions) == ["complete", "coherent", "substantive"]
    for question in questions.values():
        assert question["type"] == "noul"
        assert set(question["criteria"]) == {"true", "false"}
    assert "boilerplate" in questions["substantive"]["criteria"]["false"]
    assert "table of contents" in questions["substantive"]["criteria"]["false"]


def test_min_quality_probability_is_minimum_of_three():
    assert cv.min_quality_probability(_answers(0.9, 0.2, 0.7)) == 0.2
    with pytest.raises(KeyError):
        cv.min_quality_probability({"complete": {"noul": 0.9}})


def test_select_skips_threshold_logic():
    scores = [("a", 0.29), ("b", 0.3), ("c", 0.9), ("d", 0.95), ("e", 0.8)]
    # strictly below threshold is low quality: 0.3 is NOT below 0.3
    skipped, low, capped = cv.select_skips(
        scores, (), threshold=0.3, max_skip_fraction=0.5
    )
    assert (skipped, low, capped) == (["a"], 1, False)
    skipped, low, capped = cv.select_skips(
        scores, (), threshold=0.31, max_skip_fraction=0.5
    )
    assert skipped == ["a", "b"] and low == 2 and capped is False


def test_select_skips_never_skips_protected_parents_but_counts_them():
    scores = [("p", 0.1), ("a", 0.1), ("b", 0.9), ("c", 0.9), ("d", 0.9)]
    skipped, low, capped = cv.select_skips(
        scores, {"p"}, threshold=0.3, max_skip_fraction=0.5
    )
    assert skipped == ["a"]
    assert low == 2
    assert capped is False


def test_select_skips_cap_skips_nothing():
    scores = [("a", 0.1), ("b", 0.1), ("c", 0.9)]  # 2/3 > 0.5
    skipped, low, capped = cv.select_skips(
        scores, (), threshold=0.3, max_skip_fraction=0.5
    )
    assert skipped == [] and low == 2 and capped is True


def test_select_skips_fraction_equal_to_cap_still_skips():
    scores = [("a", 0.1), ("b", 0.9)]  # exactly 0.5, not exceeding
    skipped, low, capped = cv.select_skips(
        scores, (), threshold=0.3, max_skip_fraction=0.5
    )
    assert skipped == ["a"] and capped is False


def test_select_skips_never_skips_every_chunk():
    scores = [("a", 0.1), ("b", 0.1)]
    skipped, _, capped = cv.select_skips(
        scores, (), threshold=0.3, max_skip_fraction=1.0
    )
    assert skipped == [] and capped is True


def test_select_skips_none_scores_fail_open_and_excluded_from_denominator():
    scores = [("a", 0.1), ("b", None), ("c", None), ("d", 0.9), ("e", 0.9), ("f", 0.9)]
    skipped, low, capped = cv.select_skips(
        scores, (), threshold=0.3, max_skip_fraction=0.5
    )
    assert skipped == ["a"] and low == 1 and capped is False


def test_jev_validate_chunks_scores_every_chunk_with_one_call_each(monkeypatch, jev_settings):
    seen = {}

    def fake_many(items, max_workers=None, **kwargs):
        seen["items"] = items
        seen["max_workers"] = max_workers
        return [_answers(), _answers(substantive=0.05), _answers(), _answers()]

    monkeypatch.setattr(cv.jev, "system_one_many", fake_many)
    result = cv.jev_validate_chunks(_rows(4))

    assert [state for state, _ in seen["items"]] == [f"content {i}" for i in range(4)]
    assert all(set(q) == {"complete", "coherent", "substantive"} for _, q in seen["items"])
    assert seen["max_workers"] == 5
    assert result.validated_chunks == 4
    assert result.skipped_chunk_ids == ["c1"]
    assert result.jev_errors == 0
    assert result.cap_triggered is False
    assert result.stage_output() == {
        "backend": "jev",
        "validated_chunks": 4,
        "skipped_chunks": ["c1"],
        "low_quality_chunks": 1,
        "jev_errors": 0,
        "cap_triggered": False,
    }


def test_jev_validate_chunks_truncates_oversized_state(monkeypatch, jev_settings):
    seen = {}

    def fake_many(items, **kwargs):
        seen["state"] = items[0][0]
        return [_answers()]

    monkeypatch.setattr(cv.jev, "system_one_many", fake_many)
    cv.jev_validate_chunks([("c0", "x" * 50000)])
    assert len(seen["state"]) == cv.JEV_MAX_CHUNK_CHARS


def test_jev_validate_chunks_parent_never_skipped(monkeypatch, jev_settings):
    monkeypatch.setattr(
        cv.jev,
        "system_one_many",
        lambda items, **kw: [_answers(coherent=0.0), _answers(), _answers(), _answers()],
    )
    result = cv.jev_validate_chunks(_rows(4), protected_ids={"c0"})
    assert result.skipped_chunk_ids == []
    assert result.low_quality_chunks == 1


def test_jev_validate_chunks_cap_triggers(monkeypatch, jev_settings):
    monkeypatch.setattr(
        cv.jev,
        "system_one_many",
        lambda items, **kw: [_answers(0.0), _answers(0.0), _answers(0.0), _answers()],
    )
    result = cv.jev_validate_chunks(_rows(4))
    assert result.skipped_chunk_ids == []
    assert result.cap_triggered is True
    assert result.low_quality_chunks == 3


def test_jev_validate_chunks_failed_calls_pass_per_chunk(monkeypatch, jev_settings):
    monkeypatch.setattr(
        cv.jev,
        "system_one_many",
        lambda items, **kw: [
            RuntimeError("boom"),
            _answers(substantive=0.01),
            {"complete": {"noul": 0.9}},  # malformed answer also fails open
            _answers(),
            _answers(),
        ],
    )
    result = cv.jev_validate_chunks(_rows(5))
    assert result.skipped_chunk_ids == ["c1"]
    assert result.jev_errors == 2


def test_jev_validate_chunks_all_errors_skip_nothing(monkeypatch, jev_settings):
    monkeypatch.setattr(
        cv.jev,
        "system_one_many",
        lambda items, **kw: [RuntimeError("no credits")] * len(items),
    )
    result = cv.jev_validate_chunks(_rows(3))
    assert result.skipped_chunk_ids == []
    assert result.jev_errors == 3
    assert result.cap_triggered is False


def test_jev_validate_chunks_empty_makes_no_call(monkeypatch, jev_settings):
    def boom(*args, **kwargs):
        raise AssertionError("no Jev call expected")

    monkeypatch.setattr(cv.jev, "system_one_many", boom)
    result = cv.jev_validate_chunks([])
    assert result.validated_chunks == 0 and result.skipped_chunk_ids == []


def test_llm_validation_path_unchanged_by_jev_additions(monkeypatch):
    """validate_chunks (llm backend) never touches Jev."""
    monkeypatch.setattr("rag.chunk_validation.settings.OPENROUTER_API_KEY", "test-key")

    def boom(*args, **kwargs):
        raise AssertionError("Jev must not be used by validate_chunks")

    monkeypatch.setattr(cv.jev, "system_one_many", boom)
    chunks = [_make_chunk("Some content.", i) for i in range(5)]
    with patch("rag.chunk_validation.requests.post", return_value=_mock_pass_response()):
        assert validate_chunks(chunks, domain="general") is True
