"""Tests for WorkerSupervisor: subprocess spawn, reap, orphan reconciliation."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from rag.worker_supervisor import (
    InMemoryWorkerStore,
    WorkerInfo,
    WorkerSupervisor,
)


SLEEP_CMD = [sys.executable, "-u", "-c", "import time, sys; print('alive'); sys.stdout.flush(); time.sleep(30)"]


@pytest.fixture
def supervisor(tmp_path: Path) -> WorkerSupervisor:
    sup = WorkerSupervisor(
        store=InMemoryWorkerStore(),
        log_dir=tmp_path / "logs",
        command_builder=lambda worker_id: SLEEP_CMD,
        reap_interval=0.05,
    )
    yield sup
    sup.shutdown(timeout=2.0)


def _wait_until(predicate, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_launch_records_row_and_starts_subprocess(supervisor: WorkerSupervisor) -> None:
    ids = supervisor.launch(1)
    assert len(ids) == 1
    workers = supervisor.list()
    assert len(workers) == 1
    info = workers[0]
    assert info.id == ids[0]
    assert info.status in ("starting", "running")
    assert info.pid is not None
    assert Path(info.log_path).exists()


def test_launch_n_spawns_multiple(supervisor: WorkerSupervisor) -> None:
    ids = supervisor.launch(3)
    assert len(set(ids)) == 3
    assert len(supervisor.list()) == 3


def test_stop_terminates_subprocess(supervisor: WorkerSupervisor) -> None:
    [wid] = supervisor.launch(1)
    info_before = supervisor.get(wid)
    assert info_before.pid is not None
    supervisor.stop(wid, timeout=3.0)
    info = supervisor.get(wid)
    assert info.status in ("stopped", "crashed")
    assert info.stopped_at is not None


def test_stop_unknown_worker_raises(supervisor: WorkerSupervisor) -> None:
    with pytest.raises(KeyError):
        supervisor.stop("ghost")


def test_reap_loop_marks_crashed_on_unexpected_exit(tmp_path: Path) -> None:
    quick_exit = [sys.executable, "-c", "import sys; sys.exit(7)"]
    sup = WorkerSupervisor(
        store=InMemoryWorkerStore(),
        log_dir=tmp_path / "logs",
        command_builder=lambda wid: quick_exit,
        reap_interval=0.05,
    )
    try:
        [wid] = sup.launch(1)
        assert _wait_until(lambda: sup.get(wid).status == "crashed", timeout=3.0), sup.get(wid)
        assert sup.get(wid).exit_code == 7
    finally:
        sup.shutdown(timeout=2.0)


def test_orphan_reconciliation_marks_dead_pids_crashed(tmp_path: Path) -> None:
    store = InMemoryWorkerStore()
    # Pre-seed a "running" worker with a dead PID.
    fake = WorkerInfo(
        id="orphan-1",
        pid=999999,  # extremely unlikely to be alive
        status="running",
        started_at=time.time(),
        stopped_at=None,
        exit_code=None,
        log_path=str(tmp_path / "orphan.log"),
        host="",
    )
    store.upsert(fake)
    sup = WorkerSupervisor(store=store, log_dir=tmp_path / "logs", command_builder=lambda wid: SLEEP_CMD)
    try:
        sup.reconcile()
        assert sup.get("orphan-1").status == "crashed"
    finally:
        sup.shutdown(timeout=2.0)


def test_tail_log_yields_existing_and_new_lines(supervisor: WorkerSupervisor) -> None:
    [wid] = supervisor.launch(1)
    info = supervisor.get(wid)
    log_path = Path(info.log_path)
    assert _wait_until(lambda: log_path.exists() and log_path.stat().st_size > 0, timeout=3.0)

    gen = supervisor.tail_log(wid, follow=False)
    lines = list(gen)
    assert any("alive" in line for line in lines)


def test_shutdown_stops_all_workers(tmp_path: Path) -> None:
    sup = WorkerSupervisor(
        store=InMemoryWorkerStore(),
        log_dir=tmp_path / "logs",
        command_builder=lambda wid: SLEEP_CMD,
        reap_interval=0.05,
    )
    ids = sup.launch(2)
    sup.shutdown(timeout=3.0)
    for wid in ids:
        assert sup.get(wid).status in ("stopped", "crashed")
