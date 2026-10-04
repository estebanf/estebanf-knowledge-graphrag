import json

import httpx
import pytest

from rag import answering, opencode
from rag.answer_models import get_supported_answer_models
from rag.answering import _build_answer_prompt


def test_build_answer_prompt_emphasizes_relevance_and_gaps():
    prompt = _build_answer_prompt(
        "What does the retrieval say about cost structure?",
        {"retrieval_results": [{"chunk_id": "chunk-1", "chunk": "Evidence"}]},
    )

    assert 'User question: "What does the retrieval say about cost structure?"' in prompt
    assert "Use only the retrieved results below." in prompt
    assert (
        "Write a comprehensive narrative answer that clearly answers the question first, "
        "then elaborates on the supporting evidence." in prompt
    )
    assert "Begin with a direct thesis that answers the exact question." in prompt
    assert "Write 3 to 5 cohesive paragraphs, not bullets." in prompt
    assert "Build the answer around the most relevant evidence, not around all major themes" in prompt
    assert "Prioritize evidence that matches the narrow angle of the question." in prompt
    assert "Use broader AI-native themes only when they strengthen the insurance positioning." in prompt
    assert "Do not elevate a concept into the main answer just because it appears in a high-scoring chunk." in prompt
    assert "End with a brief statement of what the retrieval does not establish" in prompt
    assert "Do not mention chunk IDs, scores, or retrieval mechanics." in prompt
    assert "Prefer a strong narrative arc: answer, reasoning, implications, limitations." in prompt
    assert '"chunk_id": "chunk-1"' in prompt


# --- model catalog -----------------------------------------------------------


def test_exactly_one_default_answer_model_and_it_is_gpt_6_luna():
    defaults = [m for m in get_supported_answer_models() if m["default"]]
    assert [m["id"] for m in defaults] == ["gpt-6-luna"]
    assert get_supported_answer_models()[0]["id"] == "gpt-6-luna"
    flash = next(m for m in get_supported_answer_models() if m["id"] == "deepseek-v4-flash")
    assert flash["default"] is False


# --- streaming through the shared client -------------------------------------


def _stream_kwargs(model: str) -> dict:
    return dict(
        query="q",
        model=model,
        source_ids=[],
        filters={},
        seed_count=None,
        result_count=None,
        rrf_k=None,
        entity_confidence_threshold=None,
        first_hop_similarity_threshold=None,
        second_hop_similarity_threshold=None,
    )


def _install(monkeypatch, handler):
    requests = []

    def handle(request):
        requests.append(request)
        return handler(request)

    monkeypatch.setattr(answering.settings, "OPENCODE_API_KEY", "test-key")
    monkeypatch.setattr(answering.settings, "ANSWER_REASONING_EFFORT", "low")
    monkeypatch.setattr(answering, "retrieve", lambda **kw: {"retrieval_results": []})
    monkeypatch.setattr(
        opencode,
        "_new_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handle), timeout=timeout),
    )
    return requests


def _sse_body(*events) -> httpx.Response:
    text = "".join(f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, content=text.encode())


def _parse_sse(chunks: list[str]) -> list[tuple[str, dict]]:
    out = []
    for chunk in chunks:
        event, data = chunk.strip().split("\n", 1)
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_stream_answer_gpt_uses_responses_with_reasoning_and_no_temperature(monkeypatch):
    requests = _install(
        monkeypatch,
        lambda r: _sse_body(
            {"type": "response.output_text.delta", "delta": "Hello "},
            {"type": "response.output_text.delta", "delta": "world"},
            {"type": "response.completed"},
        ),
    )
    events = _parse_sse(list(answering.stream_answer(**_stream_kwargs("gpt-6-luna"))))

    assert events == [
        ("answer_delta", {"delta": "Hello "}),
        ("answer_delta", {"delta": "world"}),
        ("results", {"retrieval_results": []}),
    ]
    assert requests[0].url.path == "/zen/go/v1/responses"
    body = json.loads(requests[0].content)
    assert body["model"] == "gpt-6-luna"
    assert body["reasoning"] == {"effort": "low"}
    assert body["stream"] is True
    assert "temperature" not in body
    assert requests[0].headers[opencode.SESSION_HEADER].startswith("kgrag-answer-")


def test_stream_answer_chat_models_keep_temperature(monkeypatch):
    requests = _install(
        monkeypatch,
        lambda r: _sse_body({"choices": [{"delta": {"content": "hi"}}]}, "[DONE]"),
    )
    events = _parse_sse(list(answering.stream_answer(**_stream_kwargs("deepseek-v4-flash"))))
    assert events[0] == ("answer_delta", {"delta": "hi"})
    assert events[-1][0] == "results"
    assert requests[0].url.path == "/zen/go/v1/chat/completions"
    assert json.loads(requests[0].content)["temperature"] == 0.2


def test_stream_answer_minimax_uses_messages(monkeypatch):
    requests = _install(
        monkeypatch,
        lambda r: _sse_body(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "yo"}},
            {"type": "message_stop"},
        ),
    )
    events = _parse_sse(list(answering.stream_answer(**_stream_kwargs("minimax-m2.7"))))
    assert events[0] == ("answer_delta", {"delta": "yo"})
    assert requests[0].url.path == "/zen/go/v1/messages"
    assert json.loads(requests[0].content)["max_tokens"] == 4096


def test_stream_answer_requires_opencode_key(monkeypatch):
    _install(monkeypatch, lambda r: httpx.Response(500))
    monkeypatch.setattr(answering.settings, "OPENCODE_API_KEY", "")
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        list(answering.stream_answer(**_stream_kwargs("gpt-6-luna")))


def test_stream_answer_rejects_unsupported_model(monkeypatch):
    _install(monkeypatch, lambda r: httpx.Response(500))
    with pytest.raises(ValueError, match="Unsupported"):
        list(answering.stream_answer(**_stream_kwargs("nope-1")))
