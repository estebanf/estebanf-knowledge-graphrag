"""Tests for session-based authentication: login, logout, /me, cookie handling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api.auth import Principal, set_session_resolver
from rag.api.routes.auth import build_router, UserAuthService
from rag.config import settings


class InMemoryUserAuthService(UserAuthService):
    """In-memory user/session store usable in tests."""

    def __init__(self) -> None:
        self._users: dict[str, dict] = {}  # username -> {id, password_hash, is_active}
        self._sessions: dict[str, dict] = {}  # token -> {user_id, username, expires_at, revoked}

    def add_user(self, username: str, password: str, *, is_active: bool = True) -> str:
        user_id = f"user-{username}"
        self._users[username] = {
            "id": user_id,
            "password_hash": bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode(),
            "is_active": is_active,
        }
        return user_id

    def verify_credentials(self, username: str, password: str) -> Optional[dict]:
        rec = self._users.get(username)
        if not rec or not rec["is_active"]:
            return None
        if not bcrypt.checkpw(password.encode("utf-8"), rec["password_hash"].encode("utf-8")):
            return None
        return {"id": rec["id"], "username": username}

    def create_session(self, user_id: str, username: str, ttl_hours: int, *, ip: str = "", user_agent: str = "") -> str:
        import secrets

        token = secrets.token_urlsafe(32)
        self._sessions[token] = {
            "user_id": user_id,
            "username": username,
            "expires_at": datetime.now(timezone.utc) + timedelta(hours=ttl_hours),
            "revoked": False,
        }
        return token

    def resolve_session(self, token: str) -> Optional[dict]:
        rec = self._sessions.get(token)
        if not rec or rec["revoked"]:
            return None
        if rec["expires_at"] <= datetime.now(timezone.utc):
            return None
        return {"id": rec["user_id"], "username": rec["username"]}

    def revoke_session(self, token: str) -> None:
        if token in self._sessions:
            self._sessions[token]["revoked"] = True


@pytest.fixture
def service() -> InMemoryUserAuthService:
    return InMemoryUserAuthService()


@pytest.fixture
def app(service: InMemoryUserAuthService) -> FastAPI:
    application = FastAPI()
    application.include_router(build_router(service))
    # Wire the resolver so cookie-based principals work.
    def resolver(token: str, _req):
        info = service.resolve_session(token)
        if not info:
            return None
        return Principal(kind="session", subject=info["username"], scopes=["read", "ingest", "admin"], user_id=info["id"])

    set_session_resolver(resolver)
    yield application
    set_session_resolver(lambda _t, _r: None)


def test_login_rejects_unknown_user(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/api/auth/login", json={"username": "ghost", "password": "x"})
    assert resp.status_code == 401


def test_login_rejects_bad_password(service: InMemoryUserAuthService, app: FastAPI) -> None:
    service.add_user("demo", "good")
    client = TestClient(app)
    resp = client.post("/api/auth/login", json={"username": "demo", "password": "bad"})
    assert resp.status_code == 401


def test_login_sets_httponly_cookie(service: InMemoryUserAuthService, app: FastAPI) -> None:
    service.add_user("demo", "good")
    client = TestClient(app)
    resp = client.post("/api/auth/login", json={"username": "demo", "password": "good"})
    assert resp.status_code == 200
    assert resp.json() == {"username": "demo"}
    cookie_name = settings.RAG_SESSION_COOKIE_NAME
    set_cookie = resp.headers.get("set-cookie", "")
    assert cookie_name in set_cookie
    assert "httponly" in set_cookie.lower()
    assert "samesite=lax" in set_cookie.lower()


def test_me_returns_current_user_after_login(service: InMemoryUserAuthService, app: FastAPI) -> None:
    service.add_user("demo", "good")
    client = TestClient(app)
    client.post("/api/auth/login", json={"username": "demo", "password": "good"})
    resp = client.get("/api/auth/me")
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "demo"
    assert "scopes" in body


def test_me_returns_401_without_session(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_logout_revokes_session(service: InMemoryUserAuthService, app: FastAPI) -> None:
    service.add_user("demo", "good")
    client = TestClient(app)
    client.post("/api/auth/login", json={"username": "demo", "password": "good"})
    assert client.get("/api/auth/me").status_code == 200
    resp = client.post("/api/auth/logout")
    assert resp.status_code == 204
    assert client.get("/api/auth/me").status_code == 401


def test_inactive_user_cannot_log_in(service: InMemoryUserAuthService, app: FastAPI) -> None:
    service.add_user("disabled", "good", is_active=False)
    client = TestClient(app)
    resp = client.post("/api/auth/login", json={"username": "disabled", "password": "good"})
    assert resp.status_code == 401


def test_cookie_secure_flag_respects_setting(monkeypatch, service: InMemoryUserAuthService, app: FastAPI) -> None:
    monkeypatch.setattr(settings, "RAG_COOKIE_SECURE", True)
    service.add_user("demo", "good")
    client = TestClient(app)
    resp = client.post("/api/auth/login", json={"username": "demo", "password": "good"})
    assert "secure" in resp.headers.get("set-cookie", "").lower()
