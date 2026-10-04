"""Pipeline wiring for `PROFILING_BACKEND` and `CHUNK_VALIDATION_BACKEND`.

`execute_ingestion_pipeline` runs against a mocked connection; every external
call (LLM, Jev, parser, embedding, graph) is patched at the `rag.ingestion`
names, and network access is blocked so a missed patch fails loudly.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag import chunk_validation, jev, jev_profiling
from rag import ingestion
from rag.chunking import ChunkData
from rag.jev import JevError
from rag.profiling import DocumentProfile

MARKDOWN = "# Doc\n\nSome text.\n"

JEV_PROFILE = DocumentProfile(
    structure_type="well-structured",
    heading_consistency="consistent",
    content_density="uniform",
    primary_content_type="prose",
    avg_section_length="long",
    has_tables=True,
    has_code_blocks=False,
    domain="technical",
)
LLM_PROFILE = DocumentProfile(
    structure_type="unstructured",
    heading_consistency="none",
    content_density="uniform",
    primary_content_type="prose",
    avg_section_length="medium",
    has_tables=False,
    has_code_blocks=False,
    domain="general",
)
JEV_METADATA = {"kind": "report", "domain": "technical", "author": "Jane", "source": "Gartner"}


def _chunk(index, parent=None):
    return ChunkData(
        content=f"chunk {index}",
        token_count=2,
        chunk_index=index,
        parent_chunk_id=parent,
        chunking_strategy="recursive",
        chunking_config={},
    )


def _answers(min_p=0.9):
    return {
        "complete": {"noul": 0.95},
        "coherent": {"noul": 0.95},
        "substantive": {"noul": min_p},
    }


class Harness:
    """Runs `execute_ingestion_pipeline` with everything external mocked."""

    def __init__(
        self,
        monkeypatch,
        *,
        profiling="llm",
        validation="llm",
        stored_metadata=None,
        cached_profile=None,
        db_chunks=None,
        parent_ids=(),
        chunks=None,
    ):
        self.monkeypatch = monkeypatch
        self.completed: dict[str, dict] = {}
        self.failed: list[str] = []
        self.sql: list[tuple[str, object]] = []
        self.chunks = chunks if chunks is not None else [_chunk(i) for i in range(4)]
        self.chunk_rows = [(f"c{i}", chunk.content) for i, chunk in enumerate(self.chunks)]
        self.db_chunks = db_chunks
        self.parent_ids = list(parent_ids)

        ing = ingestion
        monkeypatch.setattr(ing.settings, "PROFILING_BACKEND", profiling)
        monkeypatch.setattr(ing.settings, "CHUNK_VALIDATION_BACKEND", validation)
        monkeypatch.setattr(ing.settings, "CHUNK_VALIDATION_JEV_THRESHOLD", 0.3)
        monkeypatch.setattr(ing.settings, "CHUNK_VALIDATION_MAX_SKIP_FRACTION", 0.5)
        monkeypatch.setattr(ing.settings, "JEV_CONCURRENCY", 2)
        monkeypatch.setattr(ing.settings, "OPENROUTER_API_KEY", "test-key")

        # Hard-block the network: any unpatched LLM/Jev call fails the test.
        def boom(*args, **kwargs):
            raise AssertionError("unexpected network call")

        monkeypatch.setattr("requests.post", boom)
        monkeypatch.setattr(jev, "_new_client", boom)

        conn = MagicMock()

        def execute(sql, params=None):
            self.sql.append((sql, params))
            result = MagicMock()
            result.fetchone.return_value = None
            result.fetchall.return_value = []
            if "SELECT s.storage_path" in sql:
                result.fetchone.return_value = ("/tmp/source.md", None, MARKDOWN)
            elif "SELECT metadata FROM sources" in sql:
                result.fetchone.return_value = (stored_metadata or {},)
            elif "stage_log -> 'parsing'" in sql:
                result.fetchone.return_value = (cached_profile,)
            elif "SELECT DISTINCT parent_chunk_id" in sql:
                result.fetchall.return_value = [(p,) for p in self.parent_ids]
            elif "SELECT id, content FROM chunks" in sql and self.db_chunks is not None:
                result.fetchall.return_value = self.db_chunks
            return result

        conn.execute.side_effect = execute
        self.conn = conn

        @contextmanager
        def fake_connection():
            yield conn

        @contextmanager
        def fake_driver():
            yield MagicMock()

        self.mocks = SimpleNamespace(
            parse_document=MagicMock(return_value=SimpleNamespace(markdown=MARKDOWN)),
            extract_metadata=MagicMock(return_value={"kind": "llm-kind", "author": None}),
            extract_profile_and_metadata=MagicMock(
                return_value=(
                    dict(JEV_METADATA),
                    SimpleNamespace(profile=JEV_PROFILE, confidence={"kind": 0.9}),
                )
            ),
            jev_classify=MagicMock(return_value=SimpleNamespace(profile=JEV_PROFILE)),
            profile_document=MagicMock(return_value=LLM_PROFILE),
            chunk_document=MagicMock(return_value=self.chunks),
            validate_chunks=MagicMock(return_value=True),
            embed_and_store_chunks=MagicMock(),
            extract_and_store_graph=MagicMock(return_value={}),
            extract_and_store_insights=MagicMock(return_value={"chunks_processed": 0}),
        )
        for name, mock in vars(self.mocks).items():
            monkeypatch.setattr(ing, name, mock)
        monkeypatch.setattr(ing, "get_connection", fake_connection)
        monkeypatch.setattr(ing, "get_graph_driver", fake_driver)
        monkeypatch.setattr(ing, "_insert_chunks", lambda *a, **k: list(self.chunk_rows))
        monkeypatch.setattr(ing, "_write_audit_log", MagicMock())
        monkeypatch.setattr(ing, "_update_stage", MagicMock())
        monkeypatch.setattr(
            ing, "_complete_stage",
            lambda conn, job_id, stage, output=None: self.completed.__setitem__(stage, output),
        )
        monkeypatch.setattr(
            ing, "_fail_stage",
            lambda conn, job_id, stage, *a, **k: self.failed.append(stage),
        )

    def jev_scores(self, per_chunk):
        """Patch Jev scoring: `per_chunk` is a list of answers (or exceptions)
        in chunk order."""
        calls = []

        def fake_many(items, **kwargs):
            calls.append(items)
            return list(per_chunk)

        self.monkeypatch.setattr(chunk_validation.jev, "system_one_many", fake_many)
        return calls

    def run(self, start_stage="parsing"):
        return ingestion.execute_ingestion_pipeline("job-1", "source-1", start_stage=start_stage)

    def soft_delete_calls(self):
        return [
            params for sql, params in self.sql
            if "UPDATE chunks SET deleted_at = now() WHERE id = ANY" in sql
        ]

    def sources_update_metadata(self):
        for sql, params in self.sql:
            if sql.startswith("UPDATE sources SET markdown_content"):
                return params[1].obj
        raise AssertionError("sources UPDATE not issued")


# ---------------------------------------------------------------------------
# T15: PROFILING_BACKEND
# ---------------------------------------------------------------------------


def test_llm_backend_profiling_path_is_unchanged(monkeypatch):
    h = Harness(monkeypatch, profiling="llm")
    h.run()

    h.mocks.extract_metadata.assert_called_once_with(MARKDOWN)
    h.mocks.profile_document.assert_called_once_with(MARKDOWN)
    h.mocks.extract_profile_and_metadata.assert_not_called()
    h.mocks.jev_classify.assert_not_called()
    assert h.completed["parsing"] == {"markdown_chars": len(MARKDOWN)}
    assert h.completed["profiling"] == {"domain": "general"}
    assert not any("stage_log -> 'parsing'" in sql for sql, _ in h.sql)
    h.mocks.chunk_document.assert_called_once_with(MARKDOWN, LLM_PROFILE)


def test_jev_backend_one_call_in_parsing_profile_persisted_and_reused(monkeypatch):
    h = Harness(monkeypatch, profiling="jev")
    h.run()

    h.mocks.extract_profile_and_metadata.assert_called_once_with(MARKDOWN)
    h.mocks.extract_metadata.assert_not_called()
    h.mocks.profile_document.assert_not_called()
    h.mocks.jev_classify.assert_not_called()  # profiling stage reuses the parsing result

    parsing = h.completed["parsing"]
    assert parsing["markdown_chars"] == len(MARKDOWN)
    assert parsing["profile_source"] == "jev"
    assert parsing["profile"] == jev_profiling.profile_to_dict(JEV_PROFILE)
    assert h.completed["profiling"] == {"domain": "technical", "profile_source": "jev"}
    h.mocks.chunk_document.assert_called_once_with(MARKDOWN, JEV_PROFILE)
    # no stage_log read needed when parsing ran in the same run
    assert not any("stage_log -> 'parsing'" in sql for sql, _ in h.sql)
    assert h.sources_update_metadata() == JEV_METADATA


def test_jev_backend_user_metadata_overrides_extracted_keys(monkeypatch):
    h = Harness(monkeypatch, profiling="jev", stored_metadata={"kind": "playbook", "team": "x"})
    h.run()

    assert h.sources_update_metadata() == {
        "kind": "playbook",  # user value wins
        "domain": "technical",
        "author": "Jane",
        "source": "Gartner",
        "team": "x",
    }


def test_jev_failure_in_parsing_falls_back_to_llm_for_metadata_and_profile(monkeypatch):
    h = Harness(monkeypatch, profiling="jev")
    h.mocks.extract_profile_and_metadata.side_effect = JevError("HTTP 500", 500)
    h.run()

    h.mocks.extract_metadata.assert_called_once_with(MARKDOWN)
    h.mocks.profile_document.assert_called_once_with(MARKDOWN)
    h.mocks.jev_classify.assert_not_called()  # no second Jev attempt in profiling
    assert h.completed["parsing"] == {"markdown_chars": len(MARKDOWN)}  # no profile persisted
    assert h.completed["profiling"] == {"domain": "general", "profile_source": "llm_fallback"}
    assert h.failed == []
    assert h.sources_update_metadata() == {"kind": "llm-kind", "author": None}


def test_retry_from_profiling_reuses_persisted_profile(monkeypatch):
    persisted = jev_profiling.profile_to_dict(JEV_PROFILE)
    h = Harness(monkeypatch, profiling="jev", cached_profile=persisted)
    h.run(start_stage="profiling")

    h.mocks.jev_classify.assert_not_called()
    h.mocks.profile_document.assert_not_called()
    h.mocks.extract_profile_and_metadata.assert_not_called()
    assert h.completed["profiling"] == {"domain": "technical", "profile_source": "jev_cached"}
    h.mocks.chunk_document.assert_called_once_with(MARKDOWN, JEV_PROFILE)


def test_retry_from_profiling_recalls_jev_when_profile_absent(monkeypatch):
    h = Harness(monkeypatch, profiling="jev", cached_profile=None)
    h.run(start_stage="profiling")

    h.mocks.jev_classify.assert_called_once_with(MARKDOWN, include_metadata=False)
    h.mocks.profile_document.assert_not_called()
    h.mocks.extract_profile_and_metadata.assert_not_called()
    assert h.completed["profiling"] == {"domain": "technical", "profile_source": "jev"}


def test_retry_from_profiling_malformed_persisted_profile_is_treated_as_absent(monkeypatch):
    h = Harness(monkeypatch, profiling="jev", cached_profile={"domain": "nonsense"})
    h.run(start_stage="profiling")
    h.mocks.jev_classify.assert_called_once()


def test_retry_from_profiling_jev_failure_falls_back_to_llm(monkeypatch):
    h = Harness(monkeypatch, profiling="jev")
    h.mocks.jev_classify.side_effect = JevError("down", 503)
    h.run(start_stage="profiling")

    h.mocks.profile_document.assert_called_once_with(MARKDOWN)
    assert h.completed["profiling"] == {"domain": "general", "profile_source": "llm_fallback"}
    assert h.failed == []


def test_llm_backend_ignores_persisted_jev_profile(monkeypatch):
    persisted = jev_profiling.profile_to_dict(JEV_PROFILE)
    h = Harness(monkeypatch, profiling="llm", cached_profile=persisted)
    h.run(start_stage="profiling")

    h.mocks.profile_document.assert_called_once_with(MARKDOWN)
    h.mocks.jev_classify.assert_not_called()
    assert h.completed["profiling"] == {"domain": "general"}


# ---------------------------------------------------------------------------
# T16: CHUNK_VALIDATION_BACKEND
# ---------------------------------------------------------------------------


def test_llm_validation_path_is_unchanged(monkeypatch):
    h = Harness(monkeypatch, validation="llm")
    h.run()

    h.mocks.validate_chunks.assert_called_once_with(h.chunks, domain="general")
    assert h.completed["validation"] == {"validated_chunks": 4}
    assert h.soft_delete_calls() == []
    embedded_rows = h.mocks.embed_and_store_chunks.call_args.args[1]
    assert embedded_rows == h.chunk_rows


def test_llm_validation_failure_still_fails_the_job(monkeypatch):
    h = Harness(monkeypatch, validation="llm")
    h.mocks.validate_chunks.return_value = False
    with pytest.raises(ValueError, match="validation failed"):
        h.run()
    assert h.failed == ["validation"]
    h.mocks.embed_and_store_chunks.assert_not_called()


def test_off_backend_skips_validation_entirely(monkeypatch):
    h = Harness(monkeypatch, validation="off")
    calls = h.jev_scores([])
    h.run()

    h.mocks.validate_chunks.assert_not_called()
    assert calls == []
    assert h.completed["validation"] == {"backend": "off", "validated_chunks": 0}
    assert h.mocks.embed_and_store_chunks.call_args.args[1] == h.chunk_rows


def test_jev_validation_scores_every_chunk_and_soft_deletes_low_quality(monkeypatch):
    h = Harness(monkeypatch, validation="jev")
    calls = h.jev_scores([_answers(), _answers(min_p=0.05), _answers(), _answers()])
    h.run()

    h.mocks.validate_chunks.assert_not_called()
    assert len(calls) == 1 and len(calls[0]) == 4  # every chunk, one call each
    assert h.soft_delete_calls() == [(["c1"],)]
    h.conn.commit.assert_called()

    expected_rows = [row for row in h.chunk_rows if row[0] != "c1"]
    assert h.mocks.embed_and_store_chunks.call_args.args[1] == expected_rows
    assert h.mocks.extract_and_store_graph.call_args.args[4] == expected_rows

    output = h.completed["validation"]
    assert output["validated_chunks"] == 4
    assert output["skipped_chunks"] == ["c1"]
    assert output["jev_errors"] == 0
    assert output["cap_triggered"] is False
    assert h.failed == []


def test_jev_validation_never_skips_parent_chunks(monkeypatch):
    h = Harness(monkeypatch, validation="jev", parent_ids=["c1"])
    h.jev_scores([_answers(), _answers(min_p=0.0), _answers(), _answers()])
    h.run()

    assert h.soft_delete_calls() == []
    assert h.mocks.embed_and_store_chunks.call_args.args[1] == h.chunk_rows
    assert h.completed["validation"]["skipped_chunks"] == []
    assert h.completed["validation"]["low_quality_chunks"] == 1


def test_jev_validation_cap_skips_nothing(monkeypatch):
    h = Harness(monkeypatch, validation="jev")
    h.jev_scores([_answers(0.0), _answers(0.0), _answers(0.0), _answers()])
    h.run()

    assert h.soft_delete_calls() == []
    assert h.mocks.embed_and_store_chunks.call_args.args[1] == h.chunk_rows
    assert h.completed["validation"]["cap_triggered"] is True
    assert h.failed == []


def test_jev_validation_fails_open_per_chunk(monkeypatch):
    h = Harness(monkeypatch, validation="jev")
    h.jev_scores([JevError("429", 429), _answers(min_p=0.0), _answers(), _answers()])
    h.run()

    assert h.soft_delete_calls() == [(["c1"],)]
    assert h.completed["validation"]["jev_errors"] == 1


def test_jev_validation_all_errors_skip_nothing_and_do_not_fail_job(monkeypatch):
    h = Harness(monkeypatch, validation="jev")
    h.jev_scores([JevError("402", 402)] * 4)
    result = h.run()

    assert result["status"] == "completed"
    assert h.soft_delete_calls() == []
    assert h.mocks.embed_and_store_chunks.call_args.args[1] == h.chunk_rows
    assert h.completed["validation"]["jev_errors"] == 4
    assert h.completed["validation"]["skipped_chunks"] == []


def test_jev_validation_retry_from_validation_scores_db_rows(monkeypatch):
    db_rows = [("d0", "alpha"), ("d1", "beta"), ("d2", "gamma"), ("d3", "delta")]
    h = Harness(monkeypatch, validation="jev", db_chunks=db_rows)
    calls = h.jev_scores([_answers(), _answers(), _answers(min_p=0.0), _answers()])
    h.run(start_stage="validation")

    assert [state for state, _ in calls[0]] == ["alpha", "beta", "gamma", "delta"]
    assert h.soft_delete_calls() == [(["d2"],)]
    assert h.mocks.embed_and_store_chunks.call_args.args[1] == [
        ("d0", "alpha"), ("d1", "beta"), ("d3", "delta"),
    ]
    h.mocks.validate_chunks.assert_not_called()
    h.mocks.chunk_document.assert_not_called()


def test_llm_validation_retry_from_validation_unchanged(monkeypatch):
    db_rows = [("d0", "alpha")]
    h = Harness(monkeypatch, validation="llm", db_chunks=db_rows)
    h.run(start_stage="validation")

    h.mocks.validate_chunks.assert_called_once_with([], domain="general")
    assert h.completed["validation"] == {"validated_chunks": 0}
    assert h.soft_delete_calls() == []


def test_jev_validation_chunks_are_soft_deleted_never_hard_deleted(monkeypatch):
    h = Harness(monkeypatch, validation="jev")
    h.jev_scores([_answers(), _answers(min_p=0.0), _answers(), _answers()])
    h.run()

    assert not any(sql.lstrip().upper().startswith("DELETE FROM CHUNKS") for sql, _ in h.sql)


def test_jev_profiling_and_jev_validation_together(monkeypatch):
    h = Harness(monkeypatch, profiling="jev", validation="jev")
    h.jev_scores([_answers(), _answers(), _answers(), _answers(min_p=0.0)])
    result = h.run()

    assert result["status"] == "completed"
    assert h.completed["profiling"]["profile_source"] == "jev"
    assert h.completed["validation"]["skipped_chunks"] == ["c3"]
