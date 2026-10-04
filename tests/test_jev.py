"""Jev (System One) client: request shape, answer parsing, retry rules and
batch fan-out. All HTTP goes through `httpx.MockTransport`."""

import json
import threading

import httpx
import pytest

from rag import jev

SLEEPS: list[float] = []


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(jev.settings, "OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setattr(jev.settings, "JEV_MODEL", "typesafe/jev-1.13")
    monkeypatch.setattr(jev.settings, "JEV_TIMEOUT_SECONDS", 20)
    monkeypatch.setattr(jev, "_sleep", lambda seconds: SLEEPS.append(seconds))
    SLEEPS.clear()


class Recorder:
    def __init__(self, monkeypatch, handler):
        self.requests: list[httpx.Request] = []
        self.handler = handler
        self._lock = threading.Lock()
        monkeypatch.setattr(
            jev,
            "_new_client",
            lambda timeout: httpx.Client(transport=httpx.MockTransport(self._handle), timeout=timeout),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
        return self.handler(request)

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


def _ok(answers: dict, **extra) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "gen-1",
            "model": "typesafe/jev-1.13",
            "provider": "typesafe",
            "answers": answers,
            "usage": {"input_tokens": 42, "output_tokens": 0, "cost": 0.000002},
            **extra,
        },
    )


def test_request_shape(monkeypatch):
    rec = Recorder(monkeypatch, lambda request: _ok({"q": {"type": "noul", "noul": 0.9}}))
    questions = {"q": jev.noul("Is it a cat?")}

    answers = jev.system_one({"text": "meow"}, questions, session_id="sess-1")

    assert answers == {"q": {"type": "noul", "noul": 0.9}}
    request = rec.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/v1/systemone"
    assert request.headers["authorization"] == "Bearer or-test-key"
    assert rec.body() == {
        "model": "typesafe/jev-1.13",
        "state": {"text": "meow"},
        "questions": questions,
        "session_id": "sess-1",
    }


def test_session_id_omitted_by_default_and_string_state(monkeypatch):
    rec = Recorder(monkeypatch, lambda request: _ok({}))
    jev.system_one("plain string state", {})
    body = rec.body()
    assert "session_id" not in body
    assert body["state"] == "plain string state"


def test_model_comes_from_settings(monkeypatch):
    monkeypatch.setattr(jev.settings, "JEV_MODEL", "typesafe/jev-9")
    rec = Recorder(monkeypatch, lambda request: _ok({}))
    jev.system_one("s", {})
    assert rec.body()["model"] == "typesafe/jev-9"


def test_timeout_default_and_override(monkeypatch):
    seen = []

    def fake_client(timeout):
        seen.append(timeout)
        return httpx.Client(
            transport=httpx.MockTransport(lambda request: _ok({})), timeout=timeout
        )

    monkeypatch.setattr(jev, "_new_client", fake_client)
    jev.system_one("s", {})
    jev.system_one("s", {}, timeout=5)
    assert seen[0].read == 20
    assert seen[1].read == 5


def test_response_exposes_usage(monkeypatch):
    Recorder(monkeypatch, lambda request: _ok({"q": {"type": "noul", "noul": 0.1}}))
    response = jev.system_one_response("s", {"q": jev.noul("x")})
    assert response.usage["input_tokens"] == 42
    assert response.id == "gen-1"
    assert response.model == "typesafe/jev-1.13"
    assert response.answers["q"]["noul"] == 0.1


def test_question_builders():
    assert jev.noul("Q") == {"type": "noul", "instructions": "Q"}
    assert jev.noul("Q", true="yes", false="no") == {
        "type": "noul",
        "instructions": "Q",
        "criteria": {"true": "yes", "false": "no"},
    }
    assert jev.noul("Q", true="yes")["criteria"] == {"true": "yes"}
    assert jev.choice("Pick", {"A": "first", "B": "second"}) == {
        "type": "choice",
        "instructions": "Pick",
        "criteria": {"A": "first", "B": "second"},
    }
    assert jev.score("Rate", ["bad", "ok", "good"]) == {
        "type": "score",
        "instructions": "Rate",
        "criteria": ["bad", "ok", "good"],
    }


def test_answer_parsing_noul_choice_score(monkeypatch):
    answers = {
        "n": {"type": "noul", "noul": 0.99},
        "c": {
            "type": "choice",
            "choice": "OPTION_A",
            "probabilities": {"OPTION_A": 0.8, "OPTION_B": 0.2},
            "confidence": 0.8,
        },
        "s": {
            "type": "score",
            "score": 1.7,
            "probabilities": [0.1, 0.2, 0.7],
            "confidence": 0.7,
        },
    }
    Recorder(monkeypatch, lambda request: _ok(answers))
    got = jev.system_one(
        "s",
        {
            "n": jev.noul("n"),
            "c": jev.choice("c", {"OPTION_A": "a", "OPTION_B": "b"}),
            "s": jev.score("s", ["l0", "l1", "l2"]),
        },
    )

    assert jev.noul_p(got["n"]) == pytest.approx(0.99)
    assert jev.choice_of(got["c"]) == "OPTION_A"
    assert jev.choice_probs(got["c"]) == {"OPTION_A": 0.8, "OPTION_B": 0.2}
    assert jev.confidence_of(got["c"]) == pytest.approx(0.8)
    assert jev.score_of(got["s"]) == pytest.approx(1.7)
    assert jev.confidence_of(got["n"]) is None


@pytest.mark.parametrize("status", [429, 529, 500, 502, 503])
def test_retries_transient_status_then_succeeds(monkeypatch, status):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(status, json={"error": "busy"})
        return _ok({"q": {"type": "noul", "noul": 0.5}})

    Recorder(monkeypatch, handler)
    assert jev.system_one("s", {"q": jev.noul("x")}) == {"q": {"type": "noul", "noul": 0.5}}
    assert calls["n"] == 3
    assert SLEEPS == [1.0, 2.0]
    assert sum(SLEEPS) < 10


def test_gives_up_after_max_attempts(monkeypatch):
    rec = Recorder(monkeypatch, lambda request: httpx.Response(529, text="overloaded"))
    with pytest.raises(jev.JevError) as exc_info:
        jev.system_one("s", {})
    assert exc_info.value.status_code == 529
    assert len(rec.requests) == 3


@pytest.mark.parametrize("status", [400, 413, 422, 401])
def test_no_retry_on_other_4xx(monkeypatch, status):
    rec = Recorder(monkeypatch, lambda request: httpx.Response(status, text="nope"))
    with pytest.raises(jev.JevError) as exc_info:
        jev.system_one("s", {})
    assert exc_info.value.status_code == status
    assert len(rec.requests) == 1
    assert SLEEPS == []


def test_402_raises_clear_credits_error(monkeypatch):
    rec = Recorder(monkeypatch, lambda request: httpx.Response(402, text="pay up"))
    with pytest.raises(jev.JevError, match="OpenRouter.*credits") as exc_info:
        jev.system_one("s", {})
    assert exc_info.value.status_code == 402
    assert len(rec.requests) == 1


def test_missing_key_raises_value_error(monkeypatch):
    monkeypatch.setattr(jev.settings, "OPENROUTER_API_KEY", "")
    rec = Recorder(monkeypatch, lambda request: _ok({}))
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        jev.system_one("s", {})
    assert rec.requests == []


def test_retry_after_honoured_and_capped(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "3"})
        if calls["n"] == 2:
            return httpx.Response(429, headers={"retry-after": "600"})
        return _ok({})

    Recorder(monkeypatch, handler)
    jev.system_one("s", {})
    # attempt 0: max(1, 3) = 3; attempt 1: max(2, 600) capped at the 4s ceiling.
    assert SLEEPS == [3.0, 4.0]
    assert sum(SLEEPS) < 10


def test_invalid_retry_after_falls_back_to_exponential(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        return _ok({})

    Recorder(monkeypatch, handler)
    jev.system_one("s", {})
    assert SLEEPS == [1.0]


def test_transport_error_retried_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        return _ok({"q": {"type": "noul", "noul": 1}})

    Recorder(monkeypatch, handler)
    assert jev.system_one("s", {})["q"]["noul"] == 1
    assert SLEEPS == [1.0]


def test_transport_error_exhausts_and_raises(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow")

    rec = Recorder(monkeypatch, handler)
    with pytest.raises(httpx.ReadTimeout):
        jev.system_one("s", {})
    assert len(rec.requests) == 3


def test_malformed_response_raises(monkeypatch):
    Recorder(monkeypatch, lambda request: httpx.Response(200, json={"id": "x"}))
    with pytest.raises(jev.JevError, match="malformed"):
        jev.system_one("s", {})


def test_system_one_many_preserves_order_and_captures_exceptions(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        state = body["state"]
        if state == "boom":
            return httpx.Response(422, text="bad")
        return _ok({"q": {"type": "noul", "noul": float(state)}})

    Recorder(monkeypatch, handler)
    items = [(str(i / 10), {"q": jev.noul("x")}) for i in range(6)]
    items.insert(2, ("boom", {"q": jev.noul("x")}))

    results = jev.system_one_many(items, max_workers=3)

    assert len(results) == 7
    assert isinstance(results[2], jev.JevError)
    assert results[2].status_code == 422
    values = [r["q"]["noul"] for i, r in enumerate(results) if i != 2]
    assert values == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_system_one_many_with_usage_returns_full_responses(monkeypatch):
    def handler(request):
        if json.loads(request.content)["state"] == "boom":
            return httpx.Response(422, text="bad")
        return _ok({"q": {"type": "noul", "noul": 0.5}})

    Recorder(monkeypatch, handler)
    results = jev.system_one_many(
        [("a", {"q": jev.noul("x")}), ("boom", {"q": jev.noul("x")})],
        max_workers=2,
        with_usage=True,
    )

    assert isinstance(results[0], jev.JevResponse)
    assert results[0].answers["q"]["noul"] == 0.5
    assert results[0].usage["cost"] == pytest.approx(0.000002)
    assert isinstance(results[1], jev.JevError)


def test_system_one_many_empty_makes_no_calls(monkeypatch):
    rec = Recorder(monkeypatch, lambda request: _ok({}))
    assert jev.system_one_many([]) == []
    assert rec.requests == []


def test_system_one_many_missing_key_is_captured_per_item(monkeypatch):
    monkeypatch.setattr(jev.settings, "OPENROUTER_API_KEY", "")
    Recorder(monkeypatch, lambda request: _ok({}))
    results = jev.system_one_many([("a", {}), ("b", {})])
    assert all(isinstance(r, ValueError) for r in results)
