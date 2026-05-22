"""Background worker supervisor.

Spawns and tracks ``rag _worker-run`` subprocesses, persists their state, and
streams their log files. The same process can be used both in production
(Postgres-backed store) and in tests (in-memory store).
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

import psycopg

from rag.config import settings
from rag.db import get_connection


WorkerStatus = str  # "starting" | "running" | "stopped" | "crashed"


@dataclass
class WorkerInfo:
    id: str
    pid: Optional[int]
    status: WorkerStatus
    started_at: float
    stopped_at: Optional[float]
    exit_code: Optional[int]
    log_path: str
    host: str


class WorkerStore(ABC):
    @abstractmethod
    def upsert(self, info: WorkerInfo) -> None: ...
    @abstractmethod
    def get(self, worker_id: str) -> Optional[WorkerInfo]: ...
    @abstractmethod
    def list(self) -> list[WorkerInfo]: ...
    @abstractmethod
    def list_active(self) -> list[WorkerInfo]: ...


class InMemoryWorkerStore(WorkerStore):
    def __init__(self) -> None:
        self._rows: dict[str, WorkerInfo] = {}
        self._lock = threading.Lock()

    def upsert(self, info: WorkerInfo) -> None:
        with self._lock:
            self._rows[info.id] = info

    def get(self, worker_id: str) -> Optional[WorkerInfo]:
        with self._lock:
            return self._rows.get(worker_id)

    def list(self) -> list[WorkerInfo]:
        with self._lock:
            return sorted(self._rows.values(), key=lambda w: w.started_at, reverse=True)

    def list_active(self) -> list[WorkerInfo]:
        with self._lock:
            return [w for w in self._rows.values() if w.status in ("starting", "running")]


def _ts_to_dt(ts: Optional[float]) -> Optional[datetime]:
    return datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None


def _dt_to_ts(dt: Optional[datetime]) -> Optional[float]:
    return dt.timestamp() if dt else None


class PostgresWorkerStore(WorkerStore):
    """Persists worker rows in the ``worker_processes`` table."""

    def upsert(self, info: WorkerInfo) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO worker_processes (id, pid, status, started_at, stopped_at, exit_code, log_path, host)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    pid = EXCLUDED.pid,
                    status = EXCLUDED.status,
                    started_at = EXCLUDED.started_at,
                    stopped_at = EXCLUDED.stopped_at,
                    exit_code = EXCLUDED.exit_code,
                    log_path = EXCLUDED.log_path,
                    host = EXCLUDED.host
                """,
                (
                    info.id, info.pid, info.status,
                    _ts_to_dt(info.started_at), _ts_to_dt(info.stopped_at),
                    info.exit_code, info.log_path, info.host,
                ),
            )
            conn.commit()

    def _row_to_info(self, row: tuple) -> WorkerInfo:
        return WorkerInfo(
            id=str(row[0]),
            pid=row[1],
            status=row[2],
            started_at=_dt_to_ts(row[3]) or 0.0,
            stopped_at=_dt_to_ts(row[4]),
            exit_code=row[5],
            log_path=row[6],
            host=row[7] or "",
        )

    def get(self, worker_id: str) -> Optional[WorkerInfo]:
        try:
            uuid.UUID(str(worker_id))
        except ValueError:
            return None
        with get_connection() as conn:
            row = conn.execute(
                "SELECT id, pid, status, started_at, stopped_at, exit_code, log_path, host FROM worker_processes WHERE id = %s",
                (worker_id,),
            ).fetchone()
        return self._row_to_info(row) if row else None

    def list(self) -> list[WorkerInfo]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, pid, status, started_at, stopped_at, exit_code, log_path, host FROM worker_processes ORDER BY started_at DESC"
            ).fetchall()
        return [self._row_to_info(r) for r in rows]

    def list_active(self) -> list[WorkerInfo]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, pid, status, started_at, stopped_at, exit_code, log_path, host FROM worker_processes WHERE status IN ('starting','running')"
            ).fetchall()
        return [self._row_to_info(r) for r in rows]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _default_command_builder(worker_id: str) -> list[str]:
    return [sys.executable, "-m", "rag.cli", "_worker-run", "--worker-id", worker_id]


class WorkerSupervisor:
    """Spawn, track, stop, tail subprocess workers."""

    def __init__(
        self,
        *,
        store: WorkerStore,
        log_dir: Path,
        command_builder: Callable[[str], list[str]] = _default_command_builder,
        reap_interval: float = 1.0,
        host: Optional[str] = None,
    ) -> None:
        self._store = store
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._command_builder = command_builder
        self._reap_interval = reap_interval
        self._host = host or socket.gethostname()
        self._procs: dict[str, subprocess.Popen] = {}
        self._log_handles: dict[str, "object"] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reaper = threading.Thread(target=self._reap_loop, name="worker-reaper", daemon=True)
        self._reaper.start()

    # --- public API ------------------------------------------------------------

    def launch(self, n: int = 1) -> list[str]:
        ids: list[str] = []
        for _ in range(max(1, n)):
            wid = str(uuid.uuid4())
            log_path = self._log_dir / f"{wid}.log"
            log_path.touch(exist_ok=True)
            info = WorkerInfo(
                id=wid,
                pid=None,
                status="starting",
                started_at=time.time(),
                stopped_at=None,
                exit_code=None,
                log_path=str(log_path),
                host=self._host,
            )
            self._store.upsert(info)
            log_fh = open(log_path, "ab", buffering=0)
            cmd = self._command_builder(wid)
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            with self._lock:
                self._procs[wid] = proc
                self._log_handles[wid] = log_fh
            info = replace(info, pid=proc.pid, status="running")
            self._store.upsert(info)
            ids.append(wid)
        return ids

    def stop(self, worker_id: str, *, timeout: float = 5.0) -> None:
        with self._lock:
            proc = self._procs.get(worker_id)
        if proc is None and self._store.get(worker_id) is None:
            raise KeyError(worker_id)
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                exit_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                exit_code = proc.wait(timeout=2.0)
        finally:
            self._finalize(worker_id, exit_code if proc.poll() is not None else proc.returncode, status="stopped")

    def get(self, worker_id: str) -> WorkerInfo:
        info = self._store.get(worker_id)
        if info is None:
            raise KeyError(worker_id)
        return info

    def list(self) -> list[WorkerInfo]:
        return self._store.list()

    def list_active(self) -> list[WorkerInfo]:
        return self._store.list_active()

    def stop_all(self, *, timeout: float = 5.0) -> list[str]:
        """Stop every active worker. Returns the IDs that were stopped."""
        stopped: list[str] = []
        with self._lock:
            ids = list(self._procs.keys())
        for wid in ids:
            try:
                self.stop(wid, timeout=timeout)
                stopped.append(wid)
            except KeyError:
                pass
        # Also reconcile any DB rows whose proc isn't tracked locally but PID is dead.
        for info in self._store.list_active():
            if info.id in stopped:
                continue
            if info.pid and not _pid_alive(info.pid):
                self._store.upsert(replace(info, status="crashed", stopped_at=time.time()))
                stopped.append(info.id)
        return stopped

    def reconcile(self) -> None:
        """Mark workers with dead PIDs as ``crashed`` (called on startup)."""
        for info in self._store.list_active():
            if info.pid and not _pid_alive(info.pid):
                self._store.upsert(replace(info, status="crashed", stopped_at=time.time()))

    def tail_log(self, worker_id: str, *, follow: bool = False, poll_interval: float = 0.5) -> Iterator[str]:
        info = self._store.get(worker_id)
        if info is None:
            raise KeyError(worker_id)
        path = Path(info.log_path)
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            # Yield existing content line-by-line.
            for line in fh:
                yield line
            if not follow:
                return
            while not self._stop_event.is_set():
                line = fh.readline()
                if line:
                    yield line
                    continue
                # Stop following when the process is no longer running.
                current = self._store.get(worker_id)
                if not current or current.status in ("stopped", "crashed"):
                    # Drain any final bytes.
                    leftover = fh.read()
                    if leftover:
                        yield leftover
                    return
                time.sleep(poll_interval)

    def shutdown(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        with self._lock:
            ids = list(self._procs.keys())
        for wid in ids:
            try:
                self.stop(wid, timeout=timeout)
            except KeyError:
                pass
        self._reaper.join(timeout=2.0)

    # --- internals -------------------------------------------------------------

    def _finalize(self, worker_id: str, exit_code: Optional[int], *, status: str) -> None:
        with self._lock:
            proc = self._procs.pop(worker_id, None)
            fh = self._log_handles.pop(worker_id, None)
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
        info = self._store.get(worker_id)
        if info is None:
            return
        # If the process exited non-zero unexpectedly, mark crashed.
        if status == "stopped" and (exit_code is not None and exit_code != 0):
            status = "crashed"
        self._store.upsert(
            replace(info, status=status, stopped_at=time.time(), exit_code=exit_code)
        )

    def _reap_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                items = list(self._procs.items())
            for wid, proc in items:
                rc = proc.poll()
                if rc is None:
                    continue
                # Exited on its own (not via stop()).
                status = "stopped" if rc == 0 else "crashed"
                self._finalize(wid, rc, status=status)
            self._stop_event.wait(self._reap_interval)


_singleton: Optional[WorkerSupervisor] = None


def get_supervisor() -> WorkerSupervisor:
    """Lazily build the production singleton (DB-backed, default log dir)."""
    global _singleton
    if _singleton is None:
        log_dir = Path(settings.RAG_WORKER_LOG_DIR)
        _singleton = WorkerSupervisor(store=PostgresWorkerStore(), log_dir=log_dir)
        _singleton.reconcile()
    return _singleton
