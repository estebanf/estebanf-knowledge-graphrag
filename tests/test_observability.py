from rag.ingestion import (
    _build_completed_stage_entry,
    _build_failed_stage_entry,
    _build_processing_stage_entry,
)
from rag.logging_config import _add_event_defaults


def test_build_processing_stage_entry_has_structured_fields():
    entry = _build_processing_stage_entry()

    assert entry["status"] == "processing"
    assert "started_at" in entry
    assert entry["error"] is None


def test_build_failed_stage_entry_includes_traceback_and_partial_output():
    entry = _build_failed_stage_entry(
        RuntimeError("boom"),
        partial_output={"chunks_written": 2},
        traceback_text="traceback...",
    )

    assert entry["status"] == "failed"
    assert entry["error"] == "boom"
    assert entry["traceback"] == "traceback..."
    assert entry["partial_output"] == {"chunks_written": 2}


def test_build_completed_stage_entry_includes_output_summary():
    entry = _build_completed_stage_entry({"chunk_count": 4})

    assert entry["status"] == "completed"
    assert "completed_at" in entry
    assert entry["output"] == {"chunk_count": 4}


def test_add_event_defaults_populates_required_logging_fields():
    event = _add_event_defaults(None, "info", {"action": "stage_end"})

    assert event["action"] == "stage_end"
    assert event["job_id"] is None
    assert event["api_key_name"] is None
    assert event["stage"] is None
    assert event["duration_ms"] == 0
    assert event["model_used"] is None
    assert event["token_count"] == 0
    assert event["status"] == "unknown"
    assert event["error"] is None
