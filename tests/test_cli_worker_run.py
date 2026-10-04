from typer.testing import CliRunner

import rag.worker
from rag.cli import app


def test_worker_run_defers_to_settings_when_options_omitted(monkeypatch):
    calls = []
    monkeypatch.setattr(rag.worker, "run_worker", lambda **kwargs: calls.append(kwargs))

    result = CliRunner().invoke(app, ["_worker-run", "--worker-id", "w1"])

    assert result.exit_code == 0, result.output
    # None lets run_worker fall back to WORKER_POLL_INTERVAL / WORKER_STUCK_JOB_MINUTES.
    assert calls == [{"poll_interval": None, "stuck_minutes": None}]


def test_worker_run_passes_explicit_options(monkeypatch):
    calls = []
    monkeypatch.setattr(rag.worker, "run_worker", lambda **kwargs: calls.append(kwargs))

    result = CliRunner().invoke(
        app, ["_worker-run", "--worker-id", "w1", "--poll-interval", "2", "--stuck-minutes", "9"]
    )

    assert result.exit_code == 0, result.output
    assert calls == [{"poll_interval": 2, "stuck_minutes": 9}]
