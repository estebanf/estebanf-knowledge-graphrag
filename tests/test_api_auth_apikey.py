"""Tests for the API key auth dependency."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import tomli_w
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from rag.api.auth import (
    KeyStore,
    Principal,
    require_api_key,
    require_principal,
    requires_scope,
)


def write_keys(path: Path, keys: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        tomli_w.dump({"keys": keys}, fh)


def make_app(store: KeyStore, *, scope: str | None = None) -> FastAPI:
    app = FastAPI()

    if scope is None:
        dep = require_api_key(store=store)
    else:
        dep = require_api_key(store=store, scope=scope)

    @app.get("/protected")
    def protected(principal: Principal = Depends(dep)) -> dict:
        return {"id": principal.subject, "scopes": principal.scopes, "kind": principal.kind}

    return app


def test_principal_dataclass_round_trip() -> None:
    p = Principal(kind="apikey", subject="cli-test", scopes=["read"])
    assert p.has_scope("read")
    assert not p.has_scope("admin")


def test_keystore_loads_keys_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "api_keys.toml"
    write_keys(path, [{"id": "cli-test", "token": "secret-abc", "scopes": ["read", "ingest"]}])
    store = KeyStore(path)
    record = store.lookup("secret-abc")
    assert record is not None
    assert record.id == "cli-test"
    assert "ingest" in record.scopes


def test_keystore_returns_none_for_unknown_token(tmp_path: Path) -> None:
    path = tmp_path / "api_keys.toml"
    write_keys(path, [{"id": "cli-test", "token": "secret-abc", "scopes": ["read"]}])
    store = KeyStore(path)
    assert store.lookup("nope") is None


def test_keystore_handles_missing_file(tmp_path: Path) -> None:
    store = KeyStore(tmp_path / "missing.toml")
    assert store.lookup("anything") is None


def test_keystore_hot_reloads_on_mtime_change(tmp_path: Path) -> None:
    path = tmp_path / "api_keys.toml"
    write_keys(path, [{"id": "cli-test", "token": "alpha", "scopes": ["read"]}])
    store = KeyStore(path)
    assert store.lookup("alpha") is not None
    assert store.lookup("beta") is None

    time.sleep(0.01)
    write_keys(path, [{"id": "cli-test", "token": "beta", "scopes": ["read"]}])
    # bump mtime explicitly in case filesystem resolution is coarse
    new_mtime = path.stat().st_mtime + 1
    import os

    os.utime(path, (new_mtime, new_mtime))
    assert store.lookup("beta") is not None
    assert store.lookup("alpha") is None


def test_require_api_key_rejects_missing_header(tmp_path: Path) -> None:
    path = tmp_path / "api_keys.toml"
    write_keys(path, [{"id": "cli", "token": "ok", "scopes": ["read"]}])
    app = make_app(KeyStore(path))
    client = TestClient(app)
    resp = client.get("/protected")
    assert resp.status_code == 401


def test_require_api_key_rejects_wrong_token(tmp_path: Path) -> None:
    path = tmp_path / "api_keys.toml"
    write_keys(path, [{"id": "cli", "token": "ok", "scopes": ["read"]}])
    app = make_app(KeyStore(path))
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_require_api_key_accepts_valid_token(tmp_path: Path) -> None:
    path = tmp_path / "api_keys.toml"
    write_keys(path, [{"id": "cli", "token": "ok", "scopes": ["read", "ingest"]}])
    app = make_app(KeyStore(path))
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer ok"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "cli"
    assert "read" in body["scopes"]
    assert body["kind"] == "apikey"


def test_require_api_key_enforces_scope(tmp_path: Path) -> None:
    path = tmp_path / "api_keys.toml"
    write_keys(path, [{"id": "cli", "token": "ok", "scopes": ["read"]}])
    app = make_app(KeyStore(path), scope="admin")
    client = TestClient(app)
    resp = client.get("/protected", headers={"Authorization": "Bearer ok"})
    assert resp.status_code == 403


def test_requires_scope_factory_returns_no_op_when_principal_has_scope() -> None:
    principal = Principal(kind="apikey", subject="cli", scopes=["admin"])
    dep = requires_scope("admin")
    # Calling the dep with the principal should not raise.
    assert dep(principal) is principal


def test_requires_scope_raises_when_principal_lacks_scope() -> None:
    principal = Principal(kind="apikey", subject="cli", scopes=["read"])
    dep = requires_scope("admin")
    with pytest.raises(HTTPException) as excinfo:
        dep(principal)
    assert excinfo.value.status_code == 403


def test_require_principal_accepts_bearer(tmp_path: Path) -> None:
    """The combined dep accepts API keys via Authorization header."""
    path = tmp_path / "api_keys.toml"
    write_keys(path, [{"id": "cli", "token": "ok", "scopes": ["read"]}])
    store = KeyStore(path)

    app = FastAPI()

    @app.get("/me")
    def me(principal: Principal = Depends(require_principal(key_store=store))) -> dict:
        return {"id": principal.subject, "kind": principal.kind}

    client = TestClient(app)
    assert client.get("/me").status_code == 401
    resp = client.get("/me", headers={"Authorization": "Bearer ok"})
    assert resp.status_code == 200
    assert resp.json() == {"id": "cli", "kind": "apikey"}
