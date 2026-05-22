"""Authentication routes: login, logout, current user.

The router takes a ``UserAuthService`` so the storage layer can be swapped (e.g.
the Postgres-backed default vs. an in-memory implementation in tests).
"""

from __future__ import annotations

import secrets
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from rag.api.auth import Principal, require_principal, set_session_resolver
from rag.config import settings
from rag.db import get_connection


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    username: str


class MeResponse(BaseModel):
    username: str
    kind: str
    scopes: list[str]


class UserAuthService(ABC):
    """Pluggable backend for users + sessions."""

    @abstractmethod
    def verify_credentials(self, username: str, password: str) -> Optional[dict]: ...

    @abstractmethod
    def create_session(
        self,
        user_id: str,
        username: str,
        ttl_hours: int,
        *,
        ip: str = "",
        user_agent: str = "",
    ) -> str: ...

    @abstractmethod
    def resolve_session(self, token: str) -> Optional[dict]: ...

    @abstractmethod
    def revoke_session(self, token: str) -> None: ...


class PostgresUserAuthService(UserAuthService):
    """Default backend: bcrypt password verification + DB-backed sessions."""

    def verify_credentials(self, username: str, password: str) -> Optional[dict]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT id, password_hash, is_active FROM users WHERE username = %s",
                (username,),
            ).fetchone()
        if not row:
            return None
        user_id, password_hash, is_active = row
        if not is_active:
            return None
        if not bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8")):
            return None
        return {"id": str(user_id), "username": username}

    def create_session(
        self,
        user_id: str,
        username: str,
        ttl_hours: int,
        *,
        ip: str = "",
        user_agent: str = "",
    ) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO user_sessions (id, user_id, expires_at, ip, user_agent) VALUES (%s, %s, %s, %s, %s)",
                (token_to_uuid(token), user_id, expires_at, ip or None, user_agent or None),
            )
            conn.execute(
                "UPDATE users SET last_login_at = now() WHERE id = %s",
                (user_id,),
            )
            conn.commit()
        return token

    def resolve_session(self, token: str) -> Optional[dict]:
        try:
            session_id = token_to_uuid(token)
        except ValueError:
            return None
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT s.user_id, u.username
                  FROM user_sessions s
                  JOIN users u ON u.id = s.user_id
                 WHERE s.id = %s
                   AND s.revoked_at IS NULL
                   AND s.expires_at > now()
                   AND u.is_active = true
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {"id": str(row[0]), "username": row[1]}

    def revoke_session(self, token: str) -> None:
        try:
            session_id = token_to_uuid(token)
        except ValueError:
            return
        with get_connection() as conn:
            conn.execute(
                "UPDATE user_sessions SET revoked_at = now() WHERE id = %s AND revoked_at IS NULL",
                (session_id,),
            )
            conn.commit()


def token_to_uuid(token: str) -> str:
    """Deterministically derive a UUID-formatted string from a random token.

    We use ``uuid5`` with a fixed namespace so the user-facing cookie value can
    be the random token (more entropy than a UUID4) while the DB stores it in
    the ``uuid`` column.
    """
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, token))


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.RAG_SESSION_COOKIE_NAME,
        value=token,
        max_age=settings.RAG_SESSION_TTL_HOURS * 3600,
        httponly=True,
        secure=bool(settings.RAG_COOKIE_SECURE),
        samesite="lax",
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=settings.RAG_SESSION_COOKIE_NAME, path="/")


def build_router(service: UserAuthService) -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["auth"])

    @router.post("/login", response_model=LoginResponse)
    def login(payload: LoginRequest, request: Request, response: Response) -> LoginResponse:
        user = service.verify_credentials(payload.username, payload.password)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
        client_ip = request.client.host if request.client else ""
        user_agent = request.headers.get("user-agent", "")
        token = service.create_session(
            user["id"],
            user["username"],
            settings.RAG_SESSION_TTL_HOURS,
            ip=client_ip,
            user_agent=user_agent,
        )
        _set_session_cookie(response, token)
        return LoginResponse(username=user["username"])

    @router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(request: Request, response: Response) -> Response:
        token = request.cookies.get(settings.RAG_SESSION_COOKIE_NAME)
        if token:
            service.revoke_session(token)
        _clear_session_cookie(response)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.get("/me", response_model=MeResponse)
    def me(principal: Principal = Depends(require_principal())) -> MeResponse:
        return MeResponse(username=principal.subject, kind=principal.kind, scopes=principal.scopes)

    return router


def install_default_session_resolver(service: UserAuthService) -> None:
    """Wire the cookie-based session resolver into ``rag.api.auth``."""

    def resolver(token: str, request: Request) -> Optional[Principal]:
        if not token:
            return None
        info = service.resolve_session(token)
        if not info:
            return None
        return Principal(
            kind="session",
            subject=info["username"],
            scopes=["read", "ingest", "admin"],
            user_id=info["id"],
        )

    set_session_resolver(resolver)
