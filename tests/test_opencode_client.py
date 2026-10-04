"""Shared OpenCode Go client: endpoint dispatch, payload rules, streaming,
retry and quota fallback. All HTTP goes through `httpx.MockTransport`."""

import json

import httpx
import pytest

from rag import opencode
from rag.opencode import SESSION_HEADER


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(opencode.settings, "OPENCODE_API_KEY", "test-key")
    monkeypatch.setattr(opencode.settings, "OPENCODE_FALLBACK_MODEL", "deepseek-v4.1-flash")
    monkeypatch.setattr(opencode, "_sleep", lambda seconds: SLEEPS.append(seconds))
    SLEEPS.clear()


SLEEPS: list[float] = []


class Recorder:
    """Install a MockTransport handler and record every request."""

    def __init__(self, monkeypatch, handler):
        self.requests: list[httpx.Request] = []
        self.handler = handler
        monkeypatch.setattr(
            opencode,
            "_new_client",
            lambda timeout: httpx.Client(transport=httpx.MockTransport(self._handle), timeout=timeout),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)

    def path(self, index: int = -1) -> str:
        return self.requests[index].url.path


def _responses_json(text: str, *, with_reasoning: bool = True, error=None) -> dict:
    output = []
    if with_reasoning:
        output.append({"type": "reasoning", "summary": []})
    output.append({"type": "message", "content": [{"type": "output_text", "text": text}]})
    return {"status": "completed", "error": error, "output": output, "usage": {}}


def _chat_json(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _sse(*events: dict | str) -> bytes:
    lines = []
    for event in events:
        data = event if isinstance(event, str) else json.dumps(event)
        lines.append(f"data: {data}\n\n")
    return "".join(lines).encode()


def _sse_response(*events) -> httpx.Response:
    return httpx.Response(200, content=_sse(*events), headers={"content-type": "text/event-stream"})


# --- endpoint dispatch -------------------------------------------------------


@pytest.mark.parametrize(
    "model,endpoint",
    [
        ("gpt-6-luna", "/responses"),
        ("minimax-m2.7", "/messages"),
        ("deepseek-v4-flash", "/chat/completions"),
        ("deepseek-v4.1-flash", "/chat/completions"),
        ("glm-5.1", "/chat/completions"),
        ("kimi-k2.6", "/chat/completions"),
        ("qwen3.6-plus", "/chat/completions"),
        ("mimo-v2.5", "/chat/completions"),
    ],
)
def test_endpoint_for_model(model, endpoint):
    assert opencode.endpoint_for_model(model) == endpoint


# --- json_call ---------------------------------------------------------------


def test_json_call_responses_skips_leading_reasoning_item(monkeypatch):
    rec = Recorder(monkeypatch, lambda r: httpx.Response(200, json=_responses_json('{"a": 1}')))
    assert opencode.json_call("gpt-6-luna", "hi") == {"a": 1}
    assert rec.path() == "/zen/go/v1/responses"


def test_json_call_responses_without_reasoning_item(monkeypatch):
    Recorder(
        monkeypatch,
        lambda r: httpx.Response(200, json=_responses_json('{"a": 2}', with_reasoning=False)),
    )
    assert opencode.json_call("gpt-6-luna", "hi") == {"a": 2}


def test_json_call_responses_concatenates_message_text_and_strips_fences(monkeypatch):
    data = {
        "status": "completed",
        "error": None,
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": '```json\n{"a"'}]},
            {"type": "message", "content": [{"type": "output_text", "text": ': 3}\n```'}]},
        ],
    }
    Recorder(monkeypatch, lambda r: httpx.Response(200, json=data))
    assert opencode.json_call("gpt-6-luna", "hi") == {"a": 3}


def test_responses_error_null_is_success_but_error_object_raises(monkeypatch):
    Recorder(monkeypatch, lambda r: httpx.Response(200, json=_responses_json('{"ok": true}', error=None)))
    assert opencode.json_call("gpt-6-luna", "hi") == {"ok": True}

    Recorder(
        monkeypatch,
        lambda r: httpx.Response(200, json=_responses_json("{}", error={"message": "boom"})),
    )
    with pytest.raises(RuntimeError, match="boom"):
        opencode.json_call("gpt-6-luna", "hi")


def test_json_call_chat_completions_parses_and_strips_fences(monkeypatch):
    rec = Recorder(monkeypatch, lambda r: httpx.Response(200, json=_chat_json('```json\n{"b": 2}\n```')))
    assert opencode.json_call("deepseek-v4-flash", "hi") == {"b": 2}
    assert rec.path() == "/zen/go/v1/chat/completions"


def test_json_call_messages_family(monkeypatch):
    rec = Recorder(
        monkeypatch,
        lambda r: httpx.Response(200, json={"content": [{"type": "text", "text": '{"m": 1}'}]}),
    )
    assert opencode.json_call("minimax-m2.7", "hi") == {"m": 1}
    assert rec.path() == "/zen/go/v1/messages"
    assert rec.body()["max_tokens"] == 4096


# --- payload rules -----------------------------------------------------------


def test_gpt_payload_has_no_temperature_and_uses_reasoning_object(monkeypatch):
    rec = Recorder(monkeypatch, lambda r: httpx.Response(200, json=_responses_json("{}")))
    opencode.json_call("gpt-6-luna", "the prompt", reasoning_effort="low")
    body = rec.body()
    assert body["model"] == "gpt-6-luna"
    assert body["input"] == "the prompt"
    assert body["reasoning"] == {"effort": "low"}
    assert body["text"] == {"format": {"type": "json_object"}}
    assert "temperature" not in body
    assert "reasoning_effort" not in body
    assert "messages" not in body


def test_gpt_payload_omits_reasoning_when_not_requested(monkeypatch):
    rec = Recorder(monkeypatch, lambda r: httpx.Response(200, json=_responses_json("{}")))
    opencode.json_call("gpt-6-luna", "p")
    assert "reasoning" not in rec.body()


def test_gpt_text_call_never_sends_temperature(monkeypatch):
    rec = Recorder(monkeypatch, lambda r: httpx.Response(200, json=_responses_json("plain")))
    assert opencode.text_call("gpt-6-luna", "p", temperature=0.5, reasoning_effort="none") == "plain"
    body = rec.body()
    assert "temperature" not in body
    assert body["reasoning"] == {"effort": "none"}
    assert "text" not in body  # no JSON mode for text calls


def test_chat_payload_uses_reasoning_effort_and_temperature_zero(monkeypatch):
    rec = Recorder(monkeypatch, lambda r: httpx.Response(200, json=_chat_json("{}")))
    opencode.json_call("deepseek-v4.1-flash", "p", reasoning_effort="none")
    body = rec.body()
    assert body["reasoning_effort"] == "none"
    assert "reasoning" not in body
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [{"role": "user", "content": "p"}]


def test_chat_payload_omits_reasoning_effort_when_not_passed(monkeypatch):
    rec = Recorder(monkeypatch, lambda r: httpx.Response(200, json=_chat_json("{}")))
    opencode.json_call("deepseek-v4-flash", "p")
    assert "reasoning_effort" not in rec.body()


def test_session_header_always_present(monkeypatch):
    rec = Recorder(monkeypatch, lambda r: httpx.Response(200, json=_chat_json("{}")))
    opencode.json_call("deepseek-v4-flash", "p", session_id="sess-1")
    assert rec.requests[-1].headers[SESSION_HEADER] == "sess-1"
    assert rec.requests[-1].headers["authorization"] == "Bearer test-key"

    opencode.json_call("deepseek-v4-flash", "p", scope="unit")
    assert rec.requests[-1].headers[SESSION_HEADER].startswith("kgrag-unit-")

    rec2 = Recorder(monkeypatch, lambda r: httpx.Response(200, json=_responses_json("{}")))
    opencode.json_call("gpt-6-luna", "p")
    assert rec2.requests[-1].headers[SESSION_HEADER]


def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr(opencode.settings, "OPENCODE_API_KEY", "")
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        opencode.json_call("deepseek-v4-flash", "p")
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        opencode.text_call("deepseek-v4-flash", "p")
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        list(opencode.stream_text("deepseek-v4-flash", "p", session_id="s"))


# --- streaming ---------------------------------------------------------------


def test_stream_responses_assembles_deltas_and_stops_on_completed(monkeypatch):
    rec = Recorder(
        monkeypatch,
        lambda r: _sse_response(
            {"type": "response.created"},
            {"type": "response.in_progress"},
            {"type": "response.output_item.added"},
            {"type": "response.content_part.added"},
            {"type": "response.output_text.delta", "delta": "Hel"},
            {"type": "response.output_text.delta", "delta": "lo"},
            {"type": "response.output_text.done", "text": "Hello"},
            {"type": "response.content_part.done"},
            {"type": "response.output_item.done"},
            {"type": "response.completed"},
            {"type": "response.output_text.delta", "delta": "IGNORED"},
        ),
    )
    out = list(opencode.stream_text("gpt-6-luna", "p", session_id="s1", reasoning_effort="low"))
    assert out == ["Hel", "lo"]
    body = rec.body()
    assert rec.path() == "/zen/go/v1/responses"
    assert body["stream"] is True
    assert body["reasoning"] == {"effort": "low"}
    assert "temperature" not in body
    assert body["max_output_tokens"] == 4096
    assert rec.requests[-1].headers[SESSION_HEADER] == "s1"


def test_stream_responses_failed_event_raises(monkeypatch):
    Recorder(
        monkeypatch,
        lambda r: _sse_response(
            {"type": "response.output_text.delta", "delta": "partial"},
            {"type": "response.failed", "response": {"error": {"message": "kaput"}}},
        ),
    )
    gen = opencode.stream_text("gpt-6-luna", "p", session_id="s")
    assert next(gen) == "partial"
    with pytest.raises(RuntimeError, match="kaput"):
        next(gen)


def test_stream_responses_error_event_raises(monkeypatch):
    Recorder(monkeypatch, lambda r: _sse_response({"type": "error", "message": "bad things"}))
    with pytest.raises(RuntimeError, match="bad things"):
        list(opencode.stream_text("gpt-6-luna", "p", session_id="s"))


def test_stream_chat_completions(monkeypatch):
    rec = Recorder(
        monkeypatch,
        lambda r: _sse_response(
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": []},
            {"choices": [{"delta": {"content": "b"}}]},
            "[DONE]",
        ),
    )
    out = list(opencode.stream_text("deepseek-v4-flash", "p", session_id="s", temperature=0.2))
    assert out == ["a", "b"]
    body = rec.body()
    assert rec.path() == "/zen/go/v1/chat/completions"
    assert body["temperature"] == 0.2
    assert body["stream"] is True
    assert "reasoning_effort" not in body


def test_stream_messages(monkeypatch):
    rec = Recorder(
        monkeypatch,
        lambda r: _sse_response(
            {"type": "message_start"},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "y"}},
            {"type": "message_stop"},
        ),
    )
    out = list(opencode.stream_text("minimax-m2.7", "p", session_id="s"))
    assert out == ["x", "y"]
    assert rec.path() == "/zen/go/v1/messages"
    assert rec.body()["max_tokens"] == 4096


# --- retry -------------------------------------------------------------------


def test_retry_on_429_then_success(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=_chat_json('{"ok": 1}'))

    Recorder(monkeypatch, handler)
    assert opencode.json_call("deepseek-v4-flash", "p") == {"ok": 1}
    assert len(calls) == 2
    assert SLEEPS == [1.0]


@pytest.mark.parametrize("status", [500, 502, 503, 529])
def test_retry_on_transient_server_statuses(monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(status)
        return httpx.Response(200, json=_chat_json("{}"))

    Recorder(monkeypatch, handler)
    assert opencode.json_call("deepseek-v4-flash", "p") == {}
    assert len(calls) == 3
    assert sum(SLEEPS) < 30


def test_retry_on_connection_error_then_success(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("slow", request=request)
        return httpx.Response(200, json=_chat_json("{}"))

    Recorder(monkeypatch, handler)
    assert opencode.json_call("deepseek-v4-flash", "p") == {}
    assert len(calls) == 2


def test_no_retry_on_400(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": "bad request"})

    Recorder(monkeypatch, handler)
    with pytest.raises(httpx.HTTPStatusError):
        opencode.json_call("deepseek-v4-flash", "p")
    assert len(calls) == 1
    assert SLEEPS == []


def test_retries_are_bounded(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    Recorder(monkeypatch, handler)
    monkeypatch.setattr(opencode.settings, "OPENCODE_FALLBACK_MODEL", "")
    with pytest.raises(httpx.HTTPStatusError):
        opencode.json_call("deepseek-v4-flash", "p")
    assert len(calls) == 3
    assert sum(SLEEPS) < 30


def test_stream_retries_before_first_delta(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503)
        return _sse_response({"choices": [{"delta": {"content": "ok"}}]}, "[DONE]")

    Recorder(monkeypatch, handler)
    assert list(opencode.stream_text("deepseek-v4-flash", "p", session_id="s")) == ["ok"]
    assert len(calls) == 2


# --- quota fallback ----------------------------------------------------------


def _quota_then_fallback(monkeypatch, status, body):
    def handler(request):
        if json.loads(request.content)["model"] == "gpt-6-luna":
            return httpx.Response(status, json=body)
        return httpx.Response(200, json=_chat_json('{"fb": true}'))

    return Recorder(monkeypatch, handler)


def test_quota_429_falls_back_after_retries(monkeypatch):
    rec = _quota_then_fallback(monkeypatch, 429, {"error": "rate"})
    assert opencode.json_call("gpt-6-luna", "p", session_id="s9") == {"fb": True}
    assert [json.loads(r.content)["model"] for r in rec.requests] == [
        "gpt-6-luna", "gpt-6-luna", "gpt-6-luna", "deepseek-v4.1-flash",
    ]
    # Fallback dispatched to its own endpoint, reasoning off, same session.
    assert rec.path() == "/zen/go/v1/chat/completions"
    assert rec.body()["reasoning_effort"] == "none"
    assert rec.requests[-1].headers[SESSION_HEADER] == "s9"


@pytest.mark.parametrize("marker", ["Quota exhausted", "monthly LIMIT reached", "Usage cap", "Budget Exceeded"])
def test_quota_4xx_body_markers_fall_back_without_retry(monkeypatch, marker):
    rec = _quota_then_fallback(monkeypatch, 402, {"error": {"message": marker}})
    assert opencode.text_call("gpt-6-luna", "p") == '{"fb": true}'
    assert len(rec.requests) == 2  # one failing attempt (4xx not retried) + fallback


def test_non_quota_400_does_not_fall_back(monkeypatch):
    rec = _quota_then_fallback(monkeypatch, 400, {"error": "malformed input"})
    with pytest.raises(httpx.HTTPStatusError):
        opencode.json_call("gpt-6-luna", "p")
    assert len(rec.requests) == 1


def test_fallback_disabled_when_empty(monkeypatch):
    monkeypatch.setattr(opencode.settings, "OPENCODE_FALLBACK_MODEL", "")
    rec = _quota_then_fallback(monkeypatch, 402, {"error": "quota exceeded"})
    with pytest.raises(httpx.HTTPStatusError):
        opencode.json_call("gpt-6-luna", "p")
    assert len(rec.requests) == 1


def test_fallback_not_used_when_same_as_requested_model(monkeypatch):
    monkeypatch.setattr(opencode.settings, "OPENCODE_FALLBACK_MODEL", "gpt-6-luna")
    rec = _quota_then_fallback(monkeypatch, 402, {"error": "quota exceeded"})
    with pytest.raises(httpx.HTTPStatusError):
        opencode.json_call("gpt-6-luna", "p")
    assert len(rec.requests) == 1


def test_fallback_happens_once_only(monkeypatch):
    rec = Recorder(monkeypatch, lambda r: httpx.Response(402, json={"error": "quota exceeded"}))
    with pytest.raises(httpx.HTTPStatusError):
        opencode.json_call("gpt-6-luna", "p")
    assert len(rec.requests) == 2  # original + one fallback attempt


def test_stream_falls_back_before_first_delta(monkeypatch):
    def handler(request):
        if json.loads(request.content)["model"] == "gpt-6-luna":
            return httpx.Response(402, json={"error": "usage limit"})
        return _sse_response({"choices": [{"delta": {"content": "fb"}}]}, "[DONE]")

    rec = Recorder(monkeypatch, handler)
    out = list(opencode.stream_text("gpt-6-luna", "p", session_id="s", reasoning_effort="low"))
    assert out == ["fb"]
    assert rec.path() == "/zen/go/v1/chat/completions"
    assert rec.body()["reasoning_effort"] == "none"


def test_stream_does_not_fall_back_after_a_delta(monkeypatch):
    class Boom(httpx.SyncByteStream):
        def __iter__(self):
            yield _sse({"choices": [{"delta": {"content": "partial"}}]})
            raise httpx.ReadError("dropped")

    rec = Recorder(monkeypatch, lambda r: httpx.Response(200, stream=Boom()))
    gen = opencode.stream_text("deepseek-v4-flash", "p", session_id="s")
    assert next(gen) == "partial"
    with pytest.raises(httpx.ReadError):
        next(gen)
    assert len(rec.requests) == 1
