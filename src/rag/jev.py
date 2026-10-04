"""Client for Jev (TypeSafe "System One") via OpenRouter.

Jev does not generate text. You send a `state` (string, JSON object or array)
plus named, typed `questions`; it answers every question in parallel in ONE
call and returns calibrated probabilities. The state is billed once per call
and each extra question adds only ~20-25 tokens, so callers should put every
question about the same state into a single call (fan-out) rather than making
one call per question.

Known jev-1.13 weaknesses worth designing around: it reads questions
literally (write precise instructions and criteria), is biased toward the first
option of a Choice (shuffle or ask twice when it matters), degrades when the
state carries lots of irrelevant text (keep the state focused), is not
resistant to adversarial content, and cannot count, do math or compare dates.
The OpenRouter context limit is 32k tokens (state + questions).

All HTTP goes through `_new_client` / `_sleep`, the same monkeypatch points as
`rag.opencode`, so tests inject an `httpx.MockTransport`.
"""

import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx

from rag.config import settings

URL = "https://openrouter.ai/api/v1/systemone"

# Transient statuses worth retrying. Every other 4xx (400/413/422 ...) is a
# caller error and is never retried; 402 (no credits) is not transient either.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 529})
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 4.0  # worst case total backoff: 2 sleeps, <= 8s

log = logging.getLogger(__name__)


class JevError(RuntimeError):
    """A Jev call failed (non-retryable status, retries exhausted, bad body)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class JevResponse:
    """Full System One response: answers plus usage for cost accounting."""

    answers: dict[str, dict]
    usage: dict = field(default_factory=dict)
    id: str | None = None
    model: str | None = None


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


def _new_client(timeout: httpx.Timeout) -> httpx.Client:
    """Single construction point for the HTTP client; tests monkeypatch this
    to inject an `httpx.MockTransport`."""
    return httpx.Client(timeout=timeout)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _require_key() -> str:
    key = settings.OPENROUTER_API_KEY
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required for Jev (System One) calls")
    return key


def _backoff_delay(attempt: int, response: httpx.Response | None) -> float:
    delay = _BACKOFF_BASE_SECONDS * (2 ** attempt)
    if response is not None:
        try:
            delay = max(delay, float(response.headers.get("retry-after", 0)))
        except ValueError:
            pass
    return min(delay, _BACKOFF_CAP_SECONDS)


def _error_detail(response: httpx.Response) -> str:
    try:
        return response.text[:300]
    except Exception:  # pragma: no cover - defensive
        return ""


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status < 400:
        return
    if status == 402:
        raise JevError(
            "Jev call failed: OpenRouter reports insufficient credits (HTTP 402). "
            "Add credits to the OpenRouter account behind OPENROUTER_API_KEY.",
            status_code=402,
        )
    raise JevError(
        f"Jev call failed with HTTP {status}: {_error_detail(response)}",
        status_code=status,
    )


def system_one_response(
    state: Any,
    questions: dict[str, dict],
    *,
    session_id: str | None = None,
    timeout: float | None = None,
) -> JevResponse:
    """Run one System One call and return the full response (answers + usage).

    Retries 429/500/502/503/529 and transport errors with exponential backoff
    (max 3 attempts, `Retry-After` honoured but capped). Raises `ValueError`
    when `OPENROUTER_API_KEY` is unset and `JevError` on every other failure.
    """
    key = _require_key()
    body: dict[str, Any] = {
        "model": settings.JEV_MODEL,
        "state": state,
        "questions": questions,
    }
    if session_id:
        body["session_id"] = session_id
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    seconds = settings.JEV_TIMEOUT_SECONDS if timeout is None else timeout
    http_timeout = httpx.Timeout(seconds, connect=min(10.0, seconds))

    with _new_client(http_timeout) as client:
        for attempt in range(_MAX_ATTEMPTS):
            last = attempt == _MAX_ATTEMPTS - 1
            try:
                response = client.post(URL, headers=headers, json=body)
            except httpx.TransportError:
                if last:
                    raise
                _sleep(_backoff_delay(attempt, None))
                continue
            if response.status_code in _RETRY_STATUSES and not last:
                _sleep(_backoff_delay(attempt, response))
                continue
            _raise_for_status(response)
            return _parse_response(response)
    raise AssertionError("unreachable")  # pragma: no cover


def _parse_response(response: httpx.Response) -> JevResponse:
    try:
        data = response.json()
        answers = data["answers"]
    except (ValueError, KeyError, TypeError) as exc:
        raise JevError(f"Jev returned a malformed response: {_error_detail(response)}") from exc
    if not isinstance(answers, dict):
        raise JevError("Jev returned a malformed response: 'answers' is not an object")
    return JevResponse(
        answers=answers,
        usage=data.get("usage") or {},
        id=data.get("id"),
        model=data.get("model"),
    )


def system_one(
    state: Any,
    questions: dict[str, dict],
    *,
    session_id: str | None = None,
    timeout: float | None = None,
) -> dict[str, dict]:
    """Run one System One call and return just the `answers` dict, keyed like
    `questions`. Use `system_one_response` when usage/cost is needed."""
    return system_one_response(
        state, questions, session_id=session_id, timeout=timeout
    ).answers


def system_one_many(
    items: Sequence[tuple[Any, dict[str, dict]]],
    *,
    max_workers: int | None = None,
    session_id: str | None = None,
    timeout: float | None = None,
) -> list[dict[str, dict] | Exception]:
    """Run many independent System One calls concurrently.

    Each item is `(state, questions)`. The result list preserves input order;
    an item whose call failed holds the exception instead of raising, so one
    bad item never discards the rest of a batch.
    """
    if not items:
        return []
    workers = max(1, min(max_workers or settings.JEV_CONCURRENCY, len(items)))

    def _run(item: tuple[Any, dict[str, dict]]) -> dict[str, dict] | Exception:
        state, questions = item
        try:
            return system_one(state, questions, session_id=session_id, timeout=timeout)
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run, items))


# ---------------------------------------------------------------------------
# Question builders and answer accessors
# ---------------------------------------------------------------------------


def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    """Yes/no question. `true`/`false` describe when each answer applies."""
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    criteria = {}
    if true is not None:
        criteria["true"] = true
    if false is not None:
        criteria["false"] = false
    if criteria:
        question["criteria"] = criteria
    return question


def choice(instructions: str, options: dict[str, str]) -> dict:
    """Pick-one question; `options` maps option key -> description (max 255)."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, levels: list[str]) -> dict:
    """Ordinal scale question; `levels[i]` describes level i (2-10 levels)."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def noul_p(answer: dict) -> float:
    """Probability of "yes" from a noul answer."""
    return float(answer["noul"])


def choice_of(answer: dict) -> str:
    """The winning option key from a choice answer."""
    return answer["choice"]


def choice_probs(answer: dict) -> dict[str, float]:
    """Per-option probabilities from a choice answer."""
    return {key: float(value) for key, value in (answer.get("probabilities") or {}).items()}


def score_of(answer: dict) -> float:
    """Probability-weighted score from a score answer."""
    return float(answer["score"])


def confidence_of(answer: dict) -> float | None:
    """Calibrated confidence for choice/score answers, if reported."""
    value = answer.get("confidence")
    return None if value is None else float(value)
