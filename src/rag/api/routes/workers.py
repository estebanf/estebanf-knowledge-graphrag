"""Worker management routes: launch, stop, list, log."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from rag.worker_supervisor import WorkerInfo, WorkerSupervisor, get_supervisor

router = APIRouter(prefix="/api/workers", tags=["workers"])


class WorkerInfoModel(BaseModel):
    id: str
    pid: Optional[int]
    status: str
    started_at: float
    stopped_at: Optional[float]
    exit_code: Optional[int]
    log_path: str
    host: str

    @classmethod
    def from_info(cls, info: WorkerInfo) -> "WorkerInfoModel":
        return cls(
            id=info.id,
            pid=info.pid,
            status=info.status,
            started_at=info.started_at,
            stopped_at=info.stopped_at,
            exit_code=info.exit_code,
            log_path=info.log_path,
            host=info.host,
        )


class WorkerListResponse(BaseModel):
    workers: list[WorkerInfoModel]


class LaunchResponse(BaseModel):
    ids: list[str]


def get_supervisor_dep() -> WorkerSupervisor:
    return get_supervisor()


@router.post("/launch", response_model=LaunchResponse)
def launch_workers(
    n: int = Query(default=1, gt=0, le=32),
    supervisor: WorkerSupervisor = Depends(get_supervisor_dep),
) -> LaunchResponse:
    ids = supervisor.launch(n)
    return LaunchResponse(ids=ids)


@router.post("/stop-all")
def stop_all_workers(
    supervisor: WorkerSupervisor = Depends(get_supervisor_dep),
) -> dict:
    stopped = supervisor.stop_all()
    return {"stopped": stopped}


@router.post("/{worker_id}/stop")
def stop_worker(
    worker_id: str,
    supervisor: WorkerSupervisor = Depends(get_supervisor_dep),
) -> dict:
    try:
        supervisor.stop(worker_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"worker not found: {worker_id}")
    return {"worker_id": worker_id, "status": "stopped"}


@router.get("", response_model=WorkerListResponse)
def list_workers(
    include_stopped: bool = Query(default=False, alias="all"),
    supervisor: WorkerSupervisor = Depends(get_supervisor_dep),
) -> WorkerListResponse:
    workers = supervisor.list() if include_stopped else supervisor.list_active()
    return WorkerListResponse(workers=[WorkerInfoModel.from_info(w) for w in workers])


@router.get("/{worker_id}/log")
def get_log(
    worker_id: str,
    follow: bool = Query(default=False),
    supervisor: WorkerSupervisor = Depends(get_supervisor_dep),
):
    try:
        # Validate existence first so we can 404 cleanly.
        supervisor.get(worker_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"worker not found: {worker_id}")

    if not follow:
        text = "".join(supervisor.tail_log(worker_id, follow=False))
        return PlainTextResponse(text)

    def event_iter():
        for line in supervisor.tail_log(worker_id, follow=True):
            yield {"event": "log", "data": line.rstrip("\n")}

    return EventSourceResponse(event_iter())
