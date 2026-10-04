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

import uuid

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
