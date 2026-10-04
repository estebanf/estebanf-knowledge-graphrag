"""Smoke tests for the read-only Jev pilot scripts. They never touch the
network, Postgres or Memgraph: connections and scoring calls are mocked."""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag import jev
from rag.profiling import DocumentProfile


def _load_script(name: str):
    script_path = Path(__file__).parent.parent / "scripts" / f"{name}.py"
    assert script_path.exists()
    spec = importlib.util.spec_from_file_location(f"{name}_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(jev, "_new_client", boom)
    monkeypatch.setattr("requests.post", boom)


def _wire_connection(module, fetchall):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.side_effect = fetchall
    module.get_connection = MagicMock()
    module.get_connection.return_value.__enter__.return_value = conn
    return conn, cur


def _assert_select_only(cur):
    statements = [call.args[0].lstrip().upper() for call in cur.execute.call_args_list]
    assert statements
    assert all(sql.startswith(("SELECT", "WITH")) for sql in statements)
    forbidden = re.compile(r"\b(INSERT|UPDATE|DELETE|TRUNCATE|DROP|ALTER)\b")
    assert not any(forbidden.search(sql) for sql in statements)


# --- pilot_jev_profiling -----------------------------------------------------


def _profile(**overrides):
    base = dict(
        structure_type="well-structured",
        heading_consistency="consistent",
        content_density="uniform",
        primary_content_type="prose",
        avg_section_length="medium",
        has_tables=False,
        has_code_blocks=False,
        domain="general",
    )
    base.update(overrides)
    return DocumentProfile(**base)


def test_pilot_profiling_no_sources(capsys):
    module = _load_script("pilot_jev_profiling")
    conn, cur = _wire_connection(module, [[]])

    assert module.main([]) == 0

    assert cur.execute.call_args.args[1] == (100,)  # --limit default
    assert conn.read_only is True
    _assert_select_only(cur)
    assert "Nothing to do" in capsys.readouterr().out


def test_pilot_profiling_reports_agreement_cost_and_latency(capsys):
    module = _load_script("pilot_jev_profiling")
    conn, cur = _wire_connection(
        module, [[("s1", "Doc one", "# a"), ("s2", "Doc two", "# b")]]
    )
    llm_profiles = {
        "# a": _profile(avg_section_length="long", domain="legal"),
        "# b": _profile(),
    }
    jev_profiles = {
        "# a": _profile(avg_section_length="long", domain="legal"),
        "# b": _profile(structure_type="unstructured", heading_consistency="none"),
    }
    module.profile_document = lambda md: llm_profiles[md]
    module.extract_metadata = lambda md: {"kind": "report", "domain": "general"}

    def fake_jev(md):
        profile = jev_profiles[md]
        return (
            {"kind": "report", "domain": "general"},
            SimpleNamespace(
                profile=profile,
                kind="report" if md == "# a" else "guide",
                metadata_domain="general",
                usage={"cost": 0.000002},
                confidence={},
            ),
        )

    module.extract_profile_and_metadata = fake_jev

    assert module.main(["--limit", "2", "--workers", "1"]) == 0

    assert cur.execute.call_args.args[1] == (2,)
    _assert_select_only(cur)
    out = capsys.readouterr().out
    assert "select_strategy(profile)  1/2 (50%)" in out  # markdown-header vs recursive
    assert "use_hierarchical          2/2 (100%)" in out
    assert "use_propositions          2/2 (100%)" in out
    assert "kind                      1/2 (50%)" in out
    assert "Jev usage cost: $0.000004" in out
    assert "Latency" in out
    assert "'Doc two'" in out  # a disagreement example


def test_pilot_profiling_summarize_handles_errors():
    module = _load_script("pilot_jev_profiling")
    rows = [
        {"source_id": "a", "name": "A", "llm_error": "x", "llm_seconds": 0.1,
         "jev_error": "y", "jev_seconds": 0.2},
    ]
    summary = module.summarize(rows)
    assert summary["compared"] == 0
    assert summary["llm_errors"] == 1 and summary["jev_errors"] == 1
    assert summary["strategy"]["rate"] is None


def test_pilot_profiling_chunking_decisions():
    module = _load_script("pilot_jev_profiling")
    decisions = module.chunking_decisions(_profile(avg_section_length="long", domain="medical"))
    assert decisions == {"strategy": "markdown-header", "hierarchical": True, "propositions": True}
    decisions = module.chunking_decisions(
        _profile(structure_type="unstructured", heading_consistency="none", domain="technical")
    )
    assert decisions == {"strategy": "recursive", "hierarchical": False, "propositions": False}


# --- pilot_jev_chunk_validation ----------------------------------------------


def _answers(min_p):
    return {
        "complete": {"noul": 0.95},
        "coherent": {"noul": 0.95},
        "substantive": {"noul": min_p},
    }


def test_pilot_chunk_validation_no_chunks(capsys):
    module = _load_script("pilot_jev_chunk_validation")
    conn, cur = _wire_connection(module, [[]])

    assert module.main([]) == 0

    assert cur.execute.call_args_list[0].args[1] == (200, 20)  # --sources default
    assert conn.read_only is True
    _assert_select_only(cur)
    assert "Nothing to do" in capsys.readouterr().out


def test_pilot_chunk_validation_end_to_end(monkeypatch, capsys):
    module = _load_script("pilot_jev_chunk_validation")
    rows = [
        ("c1", "s1", "good chunk one"),
        ("c2", "s1", "good chunk two"),
        ("c3", "s1", "Cookie banner. Subscribe now! " * 20),
        ("c4", "s2", "good chunk four"),
    ]
    conn, cur = _wire_connection(module, [rows, [("pX",)]])

    def fake_response(state, questions, **kwargs):
        assert set(questions) == {"complete", "coherent", "substantive"}
        p = 0.05 if state.startswith("Cookie") else 0.9
        return jev.JevResponse(answers=_answers(p), usage={"cost": 0.000001})

    monkeypatch.setattr(jev, "system_one_response", fake_response)

    assert module.main(["--sources", "2", "--workers", "2", "--examples", "2"]) == 0

    assert cur.execute.call_args_list[0].args[1] == (2, 20)
    assert cur.execute.call_args_list[1].args[1] == (["s1", "s2"],)
    _assert_select_only(cur)
    out = capsys.readouterr().out
    assert "chunks sampled: 4" in out
    assert "Jev usage cost: $0.000004" in out
    assert "Cookie banner" in out  # example text
    assert "..." in out  # truncated to a one-line excerpt


def test_pilot_chunk_validation_analyze_thresholds_caps_and_parents():
    module = _load_script("pilot_jev_chunk_validation")

    def row(chunk_id, source_id, min_p):
        return {
            "chunk_id": chunk_id, "source_id": source_id, "text": chunk_id,
            "min_p": min_p, "cost": 0.5,
            "probs": {"complete": 1.0, "coherent": 1.0, "substantive": min_p},
        }

    scored = [
        # s1: 1 of 4 low (0.25): skipped at 0.3, not at 0.2
        row("a", "s1", 0.25), row("b", "s1", 0.9), row("c", "s1", 0.9), row("d", "s1", 0.9),
        # s2: 3 of 4 low -> capped at every threshold they are low for
        row("e", "s2", 0.1), row("f", "s2", 0.1), row("g", "s2", 0.1), row("h", "s2", 0.9),
        # s3: low chunk is a protected parent -> never skipped
        row("p", "s3", 0.0), row("i", "s3", 0.9), row("j", "s3", 0.9),
        {"chunk_id": "z", "source_id": "s3", "text": "z", "min_p": None, "cost": 0.0, "error": "boom"},
    ]
    report = module.analyze(scored, {"p"}, max_skip_fraction=0.5)

    assert report["chunks"] == 12 and report["scored"] == 11 and report["errors"] == 1
    assert report["sources"] == 3
    assert report["cost_usd"] == pytest.approx(5.5)
    assert sum(report["histogram"]) == 11
    by_threshold = {t["threshold"]: t for t in report["thresholds"]}
    assert by_threshold[0.2]["skipped_chunks"] == 0
    assert by_threshold[0.2]["capped_sources"] == 1  # s2
    assert by_threshold[0.3]["skipped_chunks"] == 1  # only "a"
    assert by_threshold[0.3]["affected_sources"] == 1
    assert by_threshold[0.3]["capped_sources"] == 1
    assert report["lowest"][0]["min_p"] == 0.0
