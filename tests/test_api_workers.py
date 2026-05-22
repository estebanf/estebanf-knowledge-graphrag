"""Tests for /api/workers (launch, stop, list, log)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api import main as api_main
from rag.api.routes import workers as workers_route
from rag.worker_supervisor import InMemoryWorkerStore, WorkerSupervisor

SLEEP_CMD = [sys.executable, "-u", "-c", "import time, sys; print('alive'); sys.stdout.flush(); time.sleep(30)"]


@pytest.fixture
def supervisor(tmp_path: Path) -> WorkerSupervisor:
    sup = WorkerSupervisor(
        store=InMemoryWorkerStore(),
        log_dir=tmp_path / "logs",
        command_builder=lambda wid: SLEEP_CMD,
        reap_interval=0.05,
    )
    yield sup
    sup.shutdown(timeout=2.0)


@pytest.fixture
def client(supervisor: WorkerSupervisor) -> TestClient:
    app = api_main.create_app()
    app.dependency_overrides[workers_route.get_supervisor_dep] = lambda: supervisor
    return TestClient(app)


def test_launch_workers_returns_ids(client: TestClient) -> None:
    resp = client.post("/api/workers/launch?n=2")
    assert resp.status_code == 200
    body = resp.json()
    assert "ids" in body and len(body["ids"]) == 2


def test_list_workers(client: TestClient) -> None:
    client.post("/api/workers/launch?n=1")
    resp = client.get("/api/workers")
    assert resp.status_code == 200
    workers = resp.json()["workers"]
    assert len(workers) == 1
    assert workers[0]["status"] in ("starting", "running")


def test_stop_worker(client: TestClient) -> None:
    ids = client.post("/api/workers/launch?n=1").json()["ids"]
    wid = ids[0]
    resp = client.post(f"/api/workers/{wid}/stop")
    assert resp.status_code == 200
    info = client.get("/api/workers?all=true").json()["workers"][0]
    assert info["status"] in ("stopped", "crashed")


def test_stop_unknown_worker_returns_404(client: TestClient) -> None:
    assert client.post("/api/workers/ghost/stop").status_code == 404


def test_list_workers_excludes_stopped_by_default(client: TestClient) -> None:
    ids = client.post("/api/workers/launch?n=2").json()["ids"]
    client.post(f"/api/workers/{ids[0]}/stop")
    active = client.get("/api/workers").json()["workers"]
    assert len(active) == 1
    assert active[0]["id"] == ids[1]
    all_workers = client.get("/api/workers?all=true").json()["workers"]
    assert len(all_workers) == 2


def test_stop_all_workers(client: TestClient) -> None:
    ids = client.post("/api/workers/launch?n=3").json()["ids"]
    resp = client.post("/api/workers/stop-all")
    assert resp.status_code == 200
    stopped = resp.json()["stopped"]
    assert sorted(stopped) == sorted(ids)
    active = client.get("/api/workers").json()["workers"]
    assert active == []


def test_log_endpoint_returns_text_when_not_following(client: TestClient) -> None:
    ids = client.post("/api/workers/launch?n=1").json()["ids"]
    wid = ids[0]
    # Give the subprocess a beat to write its line.
    deadline = time.time() + 3
    while time.time() < deadline:
        resp = client.get(f"/api/workers/{wid}/log")
        if resp.status_code == 200 and "alive" in resp.text:
            break
        time.sleep(0.1)
    assert resp.status_code == 200
    assert "alive" in resp.text
