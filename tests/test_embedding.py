import pytest
from unittest.mock import patch, MagicMock
from rag.embedding import get_embeddings, embed_and_store_chunks

_DIM = 4096


def _mock_embedding_response(texts: list[str]) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "data": [
            {"index": i, "embedding": [0.1] * _DIM}
            for i in range(len(texts))
        ]
    }
    return mock


def test_get_embeddings_returns_vectors(monkeypatch):
    monkeypatch.setattr("rag.embedding.settings.OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("rag.embedding.settings.MODEL_EMBEDDING", "qwen/qwen3-embedding-8b")
    texts = ["hello world", "foo bar"]
    with patch("rag.embedding.requests.post", return_value=_mock_embedding_response(texts)) as mock_post:
        result = get_embeddings(texts)
    assert len(result) == 2
    assert len(result[0]) == _DIM
    mock_post.assert_called_once()


def test_get_embeddings_batches_large_inputs(monkeypatch):
    monkeypatch.setattr("rag.embedding.settings.OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("rag.embedding.settings.MODEL_EMBEDDING", "qwen/qwen3-embedding-8b")
    texts = [f"text {i}" for i in range(70)]

    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        batch = kwargs.get("json", {}).get("input", [])
        return _mock_embedding_response(batch)

    with patch("rag.embedding.requests.post", side_effect=side_effect):
        result = get_embeddings(texts)

    assert call_count == 3   # 32 + 32 + 6 = 70
    assert len(result) == 70


def test_get_embeddings_no_api_key_raises(monkeypatch):
    monkeypatch.setattr("rag.embedding.settings.OPENROUTER_API_KEY", "")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        get_embeddings(["text"])
