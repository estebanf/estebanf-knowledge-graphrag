"""Shared request plumbing for the OpenCode Go endpoints.

The provider requires an `x-opencode-session` header on every request so it can
route related calls efficiently. Requests without it are rejected outright with
HTTP 400 `MissingSessionID` (enforced from 2026-09-06), which is what took down
the `insight_extraction` stage.

The contract is "one stable ID per conversation", so an ID is minted per
logical operation — one source's insight extraction, one answer request, one
theme report — and reused for every call that operation makes, rather than
generated per HTTP request.
"""

import json
import logging
import re
import time
import uuid
from collections.abc import Iterator

import httpx

from rag.config import settings

SESSION_HEADER = "x-opencode-session"


def new_session_id(scope: str) -> str:
    """Mint one session ID for a logical operation.

    `scope` is a short human-readable tag (e.g. `insight`, `answer`) that keeps
    the IDs legible in provider-side logs when debugging routing.
    """
    return f"kgrag-{scope}-{uuid.uuid4()}"


def opencode_headers(api_key: str, session_id: str) -> dict[str, str]:
    """Standard header set for an OpenCode Go call, including the session ID."""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        SESSION_HEADER: session_id,
    }


# ---------------------------------------------------------------------------
# Shared client
#
# Every OpenCode Go caller (answers, query variants, insight extraction, theme
# reports) goes through the functions below so endpoint dispatch, payload
# rules, retry and quota fallback live in exactly one place.
# ---------------------------------------------------------------------------

BASE_URL = "https://opencode.ai/zen/go/v1"

# Transient statuses worth retrying. Every other 4xx is a caller error and is
# never retried.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 529})
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 8.0  # worst case total backoff: 1s + 2s

_QUOTA_MARKERS = ("quota", "limit", "usage", "exceeded")

log = logging.getLogger(__name__)


def endpoint_for_model(model: str) -> str:
    """Return the API path for a model family.

    gpt-* -> /responses (OpenAI Responses API); minimax-* -> /messages
    (Anthropic-style); everything else -> /chat/completions.
    """
    name = model.lower()
    if name.startswith("gpt-"):
        return "/responses"
    if name.startswith("minimax-"):
        return "/messages"
    return "/chat/completions"


def _new_client(timeout: httpx.Timeout) -> httpx.Client:
    """Single construction point for the HTTP client; tests monkeypatch this
    to inject an `httpx.MockTransport`."""
    return httpx.Client(timeout=timeout)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _timeout(seconds: float) -> httpx.Timeout:
    return httpx.Timeout(seconds, connect=10.0)


def _require_key() -> str:
    key = settings.OPENCODE_API_KEY
    if not key:
        raise ValueError("OPENCODE_API_KEY is required for OpenCode calls")
    return key


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def _build_body(
    model: str,
    prompt: str,
    *,
    json_mode: bool,
    reasoning_effort: str | None,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
) -> dict:
    endpoint = endpoint_for_model(model)
    if endpoint == "/responses":
        # `temperature` is rejected (HTTP 400) whenever reasoning is on, so it
        # is never sent to gpt-* models.
        body: dict = {"model": model, "input": prompt}
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        if max_tokens:
            body["max_output_tokens"] = max_tokens
        if json_mode:
            body["text"] = {"format": {"type": "json_object"}}
    elif endpoint == "/messages":
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or 4096,
        }
        if temperature is not None:
            body["temperature"] = temperature
    else:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            body["temperature"] = temperature
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if max_tokens:
            body["max_tokens"] = max_tokens
    if stream:
        body["stream"] = True
    return body


def _extract_text(endpoint: str, data: dict) -> str:
    """Pull the assistant text out of a non-streaming response body."""
    if endpoint == "/responses":
        if data.get("error"):
            raise RuntimeError(f"OpenCode /responses error: {data['error']}")
        if data.get("status") == "failed":
            raise RuntimeError("OpenCode /responses call failed")
        parts: list[str] = []
        for item in data.get("output") or []:
            if item.get("type") != "message":
                continue  # skip `reasoning` items that precede the message
            for block in item.get("content") or []:
                if block.get("type") == "output_text":
                    parts.append(block.get("text") or "")
        return "".join(parts)
    if endpoint == "/messages":
        return "".join(
            block.get("text") or ""
            for block in data.get("content") or []
            if block.get("type") == "text"
        )
    return data["choices"][0]["message"]["content"] or ""


def _backoff_delay(attempt: int, response: httpx.Response | None) -> float:
    delay = _BACKOFF_BASE_SECONDS * (2 ** attempt)
    if response is not None:
        try:
            delay = max(delay, float(response.headers.get("retry-after", 0)))
        except ValueError:
            pass
    return min(delay, _BACKOFF_CAP_SECONDS)


def _raise_for_status(response: httpx.Response) -> None:
    # A streamed response's body must be read before `.text` is usable on the
    # resulting HTTPStatusError (quota detection inspects it).
    if response.is_error:
        response.read()
    response.raise_for_status()


def _is_quota_error(exc: BaseException) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    status = exc.response.status_code
    if status == 429:
        return True
    if 400 <= status < 500:
        try:
            body = exc.response.text.lower()
        except httpx.ResponseNotRead:
            return False
        return any(marker in body for marker in _QUOTA_MARKERS)
    return False


def _fallback_for(model: str, exc: BaseException) -> str | None:
    fallback = (settings.OPENCODE_FALLBACK_MODEL or "").strip()
    if not fallback or fallback == model or not _is_quota_error(exc):
        return None
    log.warning(
        "OpenCode model %s failed (status %s); falling back to %s",
        model,
        getattr(getattr(exc, "response", None), "status_code", "?"),
        fallback,
    )
    return fallback


def _post_once(
    model: str,
    prompt: str,
    *,
    session_id: str,
    json_mode: bool,
    reasoning_effort: str | None,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float,
) -> str:
    """One model, bounded retry. Returns the response text."""
    endpoint = endpoint_for_model(model)
    headers = opencode_headers(_require_key(), session_id)
    body = _build_body(
        model, prompt, json_mode=json_mode, reasoning_effort=reasoning_effort,
        temperature=temperature, max_tokens=max_tokens, stream=False,
    )
    with _new_client(_timeout(timeout)) as client:
        for attempt in range(_MAX_ATTEMPTS):
            last = attempt == _MAX_ATTEMPTS - 1
            try:
                response = client.post(BASE_URL + endpoint, headers=headers, json=body)
            except httpx.TransportError:
                if last:
                    raise
                _sleep(_backoff_delay(attempt, None))
                continue
            if response.status_code in _RETRY_STATUSES and not last:
                _sleep(_backoff_delay(attempt, response))
                continue
            _raise_for_status(response)
            return _extract_text(endpoint, response.json())
    raise AssertionError("unreachable")  # pragma: no cover


def _call_text(
    model: str,
    prompt: str,
    *,
    session_id: str,
    json_mode: bool,
    reasoning_effort: str | None,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float,
) -> str:
    kwargs = dict(
        session_id=session_id, json_mode=json_mode, max_tokens=max_tokens, timeout=timeout,
    )
    try:
        return _post_once(
            model, prompt, reasoning_effort=reasoning_effort, temperature=temperature, **kwargs
        )
    except httpx.HTTPStatusError as exc:
        fallback = _fallback_for(model, exc)
        if fallback is None:
            raise
        return _post_once(
            fallback, prompt, reasoning_effort="none", temperature=temperature, **kwargs
        )


def json_call(
    model: str,
    prompt: str,
    *,
    session_id: str | None = None,
    reasoning_effort: str | None = None,
    timeout: float = 90,
    scope: str = "call",
    max_tokens: int | None = None,
) -> dict:
    """One-shot call that returns parsed JSON.

    gpt-* models use `/responses` with `text.format=json_object`; other
    chat-completions models use `response_format=json_object` at temperature 0.
    On quota exhaustion the call is retried once on `OPENCODE_FALLBACK_MODEL`.
    """
    _require_key()
    text = _call_text(
        model,
        prompt,
        session_id=session_id or new_session_id(scope),
        json_mode=True,
        reasoning_effort=reasoning_effort,
        temperature=0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return json.loads(_strip_fences(text))


def text_call(
    model: str,
    prompt: str,
    *,
    session_id: str | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    timeout: float = 120,
    scope: str = "call",
    max_tokens: int | None = None,
) -> str:
    """One-shot plain-text call (same retry and quota-fallback rules)."""
    _require_key()
    text = _call_text(
        model,
        prompt,
        session_id=session_id or new_session_id(scope),
        json_mode=False,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return text.strip()


def _iter_stream_deltas(endpoint: str, lines: Iterator[str]) -> Iterator[str]:
    """Parse an SSE line stream into text deltas for the given endpoint."""
    for raw_line in lines:
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if endpoint == "/chat/completions" and data == "[DONE]":
            return
        payload = json.loads(data)
        kind = payload.get("type")
        if endpoint == "/responses":
            if kind == "response.output_text.delta":
                if payload.get("delta"):
                    yield payload["delta"]
            elif kind == "response.completed":
                return
            elif kind == "response.failed":
                error = (payload.get("response") or {}).get("error") or payload.get("error")
                raise RuntimeError(f"OpenCode /responses stream failed: {error}")
            elif kind == "error":
                raise RuntimeError(f"OpenCode /responses stream error: {payload.get('message') or payload}")
        elif endpoint == "/messages":
            if kind == "content_block_delta":
                text = (payload.get("delta") or {}).get("text")
                if text:
                    yield text
            elif kind == "message_stop":
                return
            elif kind == "error":
                raise RuntimeError(f"OpenCode /messages stream error: {payload.get('error') or payload}")
        else:
            if payload.get("error"):
                raise RuntimeError(f"OpenCode stream error: {payload['error']}")
            choices = payload.get("choices") or [{}]
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta


def _stream_once(
    model: str,
    prompt: str,
    *,
    session_id: str,
    reasoning_effort: str | None,
    temperature: float | None,
    max_tokens: int,
    timeout: float,
) -> Iterator[str]:
    """One model, bounded retry of the connect phase. Once a delta has been
    yielded a failure propagates; the stream is not restarted."""
    endpoint = endpoint_for_model(model)
    headers = opencode_headers(_require_key(), session_id)
    body = _build_body(
        model, prompt, json_mode=False, reasoning_effort=reasoning_effort,
        temperature=temperature, max_tokens=max_tokens, stream=True,
    )
    with _new_client(_timeout(timeout)) as client:
        for attempt in range(_MAX_ATTEMPTS):
            last = attempt == _MAX_ATTEMPTS - 1
            yielded = False
            try:
                with client.stream("POST", BASE_URL + endpoint, headers=headers, json=body) as response:
                    if response.status_code in _RETRY_STATUSES and not last:
                        response.read()
                        _sleep(_backoff_delay(attempt, response))
                        continue
                    _raise_for_status(response)
                    for delta in _iter_stream_deltas(endpoint, response.iter_lines()):
                        yielded = True
                        yield delta
                    return
            except httpx.TransportError:
                if yielded or last:
                    raise
                _sleep(_backoff_delay(attempt, None))


def stream_text(
    model: str,
    prompt: str,
    *,
    session_id: str,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 4096,
    timeout: float = 300,
) -> Iterator[str]:
    """Yield text deltas for any model family.

    Quota fallback applies only when the failure happens before the first
    delta has been yielded.
    """
    kwargs = dict(session_id=session_id, max_tokens=max_tokens, timeout=timeout)
    yielded = False
    try:
        for delta in _stream_once(
            model, prompt, reasoning_effort=reasoning_effort, temperature=temperature, **kwargs
        ):
            yielded = True
            yield delta
    except httpx.HTTPStatusError as exc:
        fallback = None if yielded else _fallback_for(model, exc)
        if fallback is None:
            raise
        yield from _stream_once(
            fallback, prompt, reasoning_effort="none", temperature=temperature, **kwargs
        )
