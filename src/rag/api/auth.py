"""API authentication: file-based API keys (Bearer) and DB-backed user sessions.

Two principal kinds:
- ``apikey``: validated against ``data/api_keys.toml`` (hot-reload on mtime change)
- ``session``: validated against the ``user_sessions`` table (Phase 4)

Routes use ``require_principal()`` to accept either. ``requires_scope("admin")``
adds scope enforcement on top of an already-resolved principal.
"""

from __future__ import annotations

import logging
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, Header, HTTPException, Request, status

from rag.config import settings

log = logging.getLogger(__name__)


# --- Principal & API key store --------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """A resolved authenticated caller.

    ``kind`` is ``"apikey"`` or ``"session"``. ``subject`` identifies the caller
    (key id for API keys, username for sessions). ``scopes`` is a flat list of
    permission strings (``"read"``, ``"ingest"``, ``"admin"``).
    """

    kind: str
    subject: str
    scopes: list[str] = field(default_factory=list)
    user_id: Optional[str] = None

    def has_scope(self, scope: str) -> bool:
        return "admin" in self.scopes or scope in self.scopes


@dataclass(frozen=True)
class KeyRecord:
    id: str
    token: str
    scopes: list[str]


class KeyStore:
    """File-backed API key store with mtime-based hot reload.

    The store keeps an in-memory dict keyed by the bearer token. It rechecks the
    file's mtime on each lookup; any change triggers a reload. Lookups are O(1)
    and thread-safe.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._mtime: float = -1.0
        self._by_token: dict[str, KeyRecord] = {}
        self._reload_if_changed()

    @property
    def path(self) -> Path:
        return self._path

    def _reload_if_changed(self) -> None:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            with self._lock:
                if self._mtime != 0.0 or self._by_token:
                    self._mtime = 0.0
                    self._by_token = {}
            return
        mtime = stat.st_mtime
        if mtime == self._mtime:
            return
        with self._lock:
            if mtime == self._mtime:
                return
            try:
                with self._path.open("rb") as fh:
                    data = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                log.warning("api_keys file unreadable at %s: %s", self._path, exc)
                self._mtime = mtime
                self._by_token = {}
                return
            new: dict[str, KeyRecord] = {}
            for entry in data.get("keys", []):
                token = entry.get("token")
                key_id = entry.get("id")
                if not token or not key_id:
                    continue
                scopes = list(entry.get("scopes") or [])
                new[token] = KeyRecord(id=key_id, token=token, scopes=scopes)
            self._mtime = mtime
            self._by_token = new
            log.info("loaded %d api keys from %s", len(new), self._path)

    def lookup(self, token: str) -> Optional[KeyRecord]:
        self._reload_if_changed()
        return self._by_token.get(token)

    def all_ids(self) -> list[str]:
        self._reload_if_changed()
        return sorted({rec.id for rec in self._by_token.values()})


_default_store: Optional[KeyStore] = None


def get_default_keystore() -> KeyStore:
    global _default_store
    if _default_store is None:
        _default_store = KeyStore(Path(settings.RAG_API_KEYS_PATH))
    return _default_store


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


# --- FastAPI dependencies -------------------------------------------------------


def require_api_key(
    store: Optional[KeyStore] = None,
    *,
    scope: Optional[str] = None,
) -> Callable[..., Principal]:
    """Dependency factory: validate an Authorization: Bearer token."""

    def dep(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ) -> Principal:
        keystore = store or get_default_keystore()
        token = _extract_bearer(authorization)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        record = keystore.lookup(token)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid api key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if scope is not None and not _scopes_include(record.scopes, scope):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient scope")
        return Principal(kind="apikey", subject=record.id, scopes=list(record.scopes))

    return dep


def _scopes_include(scopes: list[str], required: str) -> bool:
    return "admin" in scopes or required in scopes


def requires_scope(scope: str) -> Callable[..., Principal]:
    """Dependency that enforces a scope on top of the shared principal dependency.

    Resolves the principal through ``default_principal_dependency`` so a route
    can require a scope with a single ``Depends(requires_scope("ingest"))``. Using
    the shared singleton keeps FastAPI's per-request dependency cache and the test
    auth-bypass override (``app.dependency_overrides[default_principal_dependency]``)
    in effect, rather than triggering a second, un-overridable auth resolution.
    """

    def dep(principal: Principal = Depends(default_principal_dependency)) -> Principal:
        if not principal.has_scope(scope):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient scope")
        return principal

    return dep


# --- Session resolver (Phase 4 fills in real DB lookup) -------------------------

# Placeholder type used by ``require_principal`` so Phase 4 can plug in a real
# session resolver without changing call sites.
SessionResolver = Callable[[str, Request], Optional[Principal]]


def _null_session_resolver(_token: str, _request: Request) -> Optional[Principal]:
    return None


_session_resolver: SessionResolver = _null_session_resolver


def set_session_resolver(resolver: SessionResolver) -> None:
    """Install the session-cookie resolver (called by Phase 4 wiring)."""
    global _session_resolver
    _session_resolver = resolver


def require_principal(
    *,
    key_store: Optional[KeyStore] = None,
    scope: Optional[str] = None,
) -> Callable[..., Principal]:
    """Accept either an API key (Bearer) or a session cookie."""

    def dep(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ) -> Principal:
        keystore = key_store or get_default_keystore()
        token = _extract_bearer(authorization)
        if token:
            record = keystore.lookup(token)
            if record is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="invalid api key",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            principal = Principal(kind="apikey", subject=record.id, scopes=list(record.scopes))
        else:
            cookie_name = settings.RAG_SESSION_COOKIE_NAME
            session_token = request.cookies.get(cookie_name)
            principal = _session_resolver(session_token, request) if session_token else None
            if principal is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        if scope is not None and not principal.has_scope(scope):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient scope")
        return principal

    return dep


# Shared principal dependency reused across the app so FastAPI's per-request
# dependency cache and the test auth-bypass override key on one stable object.
# ``main.py`` binds ``principal_dep`` to this and ``requires_scope`` sub-depends
# on it. Instantiate once — ``require_principal()`` returns a fresh closure per
# call, and per-request caching + dependency_overrides both key on identity.
default_principal_dependency = require_principal()
