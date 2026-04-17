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
