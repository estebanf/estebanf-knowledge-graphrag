"""Jev canonical-name correction (`rag.entity_names`): question construction,
the order-agreement + confidence gate, rename-vs-merge planning, alias
bookkeeping, dry-run safety. Offline: Jev, Postgres and Memgraph are doubles."""
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audit_fakes import FakeConn, FakeSession
from rag import entity_names as en
from rag import jev
from rag.jev import JevError, JevResponse

NOW = datetime(2026, 1, 1)


def _load_semantic():
    path = Path(__file__).parent.parent / "scripts" / "merge_semantic_duplicates.py"
    spec = importlib.util.spec_from_file_location("merge_semantic_for_names", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PICK = _load_semantic().pick_survivor


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(jev, "_new_client", boom)
    monkeypatch.setattr(en.settings, "OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr(en.settings, "ENTITY_NAME_MIN_CONFIDENCE", 0.8)
    monkeypatch.setattr(en.settings, "ENTITY_NAME_REVIEW_MIN_ALIASES", 3)


def _entity(eid, name, aliases=(), etype="ORGANIZATION", emb=True, age_days=0):
    return {
        "id": eid,
        "canonical_name": name,
        "entity_type": etype,
        "aliases": list(aliases),
        "has_embedding": emb,
        "created_at": NOW - timedelta(days=age_days),
    }


def _row(e):
    return (e["id"], e["canonical_name"], e["entity_type"], e["aliases"], e["has_embedding"], e["created_at"])


def _vote(keys, winner, p=0.9):
    rest = [k for k in keys if k != winner]
    probs = {winner: p}
    for k in rest:
        probs[k] = (1 - p) / max(1, len(rest))
    return {"type": "choice", "choice": winner, "probabilities": probs, "confidence": p}


def _answers(names, winner_a, winner_b=None, p=0.9, p_b=None):
    keys = en.option_keys(names)
    return {
        "name__a": _vote(keys, f"n{winner_a}", p),
        "name__b": _vote(keys, f"n{winner_a if winner_b is None else winner_b}", p if p_b is None else p_b),
    }


def _fake_many(per_state, calls=None):
    """`per_state(state, questions)` -> answers dict or Exception."""

    def fake(items, **kwargs):
        out = []
        for state, questions in items:
            if calls is not None:
                calls.append((state, questions))
            result = per_state(state, questions)
            out.append(result if isinstance(result, Exception) else JevResponse(answers=result, usage={"cost": 0.0001}))
        return out

    return fake


# ── Question construction ────────────────────────────────────────────────────


def test_candidate_names_dedupes_keeps_canonical_first_and_caps():
    names = en.candidate_names("Open AI", ["OpenAI", "Open AI", " ", "OpenAI", "openai", ""])
    assert names == ["Open AI", "OpenAI", "openai"]  # case-sensitive: casing is part of a spelling

    many = [f"alias {i}" for i in range(300)]
    capped = en.candidate_names("Canon", many)
    assert len(capped) == en.MAX_NAMES == 120
    assert capped[0] == "Canon"


def test_name_questions_use_safe_n_keys_and_a_reversed_duplicate():
    names = ["Open AI", "OpenAI", 'Open "AI" Inc', "Open-AI"]
    state = en.build_name_state("ORGANIZATION", names)
    questions = en.build_name_questions(names)

    assert state == {"entity_type": "ORGANIZATION", "names": names}
    assert set(questions) == {"name__a", "name__b"}
    a, b = questions["name__a"], questions["name__b"]
    assert a["type"] == b["type"] == "choice"
    assert a["instructions"] == b["instructions"]
    assert "correct, standard real-world spelling" in a["instructions"]
    assert list(a["criteria"]) == ["n0", "n1", "n2", "n3"]
    assert list(b["criteria"]) == ["n3", "n2", "n1", "n0"]
    # the key always denotes the same string, whatever the order; odd characters
    # live only in the description, never in the key
    assert a["criteria"] == b["criteria"] == {f"n{i}": n for i, n in enumerate(names)}


# ── Agreement + confidence gate ──────────────────────────────────────────────

NAMES = ["Open AI", "OpenAI", "open ai"]


def test_rename_requires_both_orders_to_agree_and_confidence_to_clear():
    status, winner, conf = en.evaluate_answers(_answers(NAMES, 1, p=0.92), NAMES, 0.8)
    assert (status, winner, conf) == (en.STATUS_RENAME, "OpenAI", 0.92)

    # the two orders pick different options: no action, whatever the probabilities
    status, winner, _ = en.evaluate_answers(_answers(NAMES, 1, winner_b=2, p=0.95), NAMES, 0.8)
    assert (status, winner) == (en.STATUS_UNCERTAIN, None)

    # both agree but the averaged probability is below the threshold
    status, winner, conf = en.evaluate_answers(_answers(NAMES, 1, p=0.7, p_b=0.8), NAMES, 0.8)
    assert (status, winner, conf) == (en.STATUS_UNCERTAIN, None, 0.75)

    # the average (not each order alone) is what is compared with the threshold
    status, winner, conf = en.evaluate_answers(_answers(NAMES, 1, p=0.9, p_b=0.7), NAMES, 0.8)
    assert (status, winner, conf) == (en.STATUS_RENAME, "OpenAI", 0.8)


def test_confident_vote_for_the_current_name_is_keep_not_a_rename():
    status, winner, _ = en.evaluate_answers(_answers(NAMES, 0, p=0.95), NAMES, 0.8)
    assert (status, winner) == (en.STATUS_KEEP, "Open AI")


def test_unusable_answers_are_uncertain():
    assert en.evaluate_answers({}, NAMES, 0.8)[0] == en.STATUS_UNCERTAIN
    only_a = {"name__a": _answers(NAMES, 1)["name__a"]}
    assert en.evaluate_answers(only_a, NAMES, 0.8)[0] == en.STATUS_UNCERTAIN
    bad = {"name__a": {"choice": "zzz"}, "name__b": {"choice": "zzz"}}
    assert en.evaluate_answers(bad, NAMES, 0.8)[0] == en.STATUS_UNCERTAIN


def test_judge_names_sends_one_call_per_entity_and_isolates_jev_errors(monkeypatch):
    hub = _entity("e1", "Open AI", ["OpenAI", "Open-AI", "OpenAI Inc"])
    broken = _entity("e2", "Antropic", ["Anthropic", "Anthropic PBC", "Antropic AI"])
    single = _entity("e3", "Solo", ["Solo", "Solo"])  # one distinct name: nothing to choose between
    calls = []

    def per_state(state, questions):
        if state["names"][0] == "Antropic":
            return JevError("boom", status_code=500)
        return _answers(state["names"], 1)

    monkeypatch.setattr(jev, "system_one_many", _fake_many(per_state, calls))

    judgements, stats = en.judge_names([hub, broken, single], progress=lambda m: None)

    assert len(calls) == 2  # `single` never reached Jev
    assert calls[0][0] == {"entity_type": "ORGANIZATION", "names": ["Open AI", "OpenAI", "Open-AI", "OpenAI Inc"]}
    by_id = {j.entity["id"]: j for j in judgements}
    assert (by_id["e1"].status, by_id["e1"].winner) == (en.STATUS_RENAME, "OpenAI")
    assert by_id["e2"].status == en.STATUS_ERROR and "boom" in by_id["e2"].error
    assert by_id["e3"].status == en.STATUS_KEEP
    assert stats.jev_errors == 1 and stats.targets == 3


def test_judge_names_without_api_key_makes_no_call_and_changes_nothing(monkeypatch):
    monkeypatch.setattr(en.settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(jev, "system_one_many", MagicMock(side_effect=AssertionError("no call expected")))

    judgements, stats = en.judge_names(
        [_entity("e1", "Open AI", ["OpenAI", "Open-AI", "OpenAI Inc"])], progress=lambda m: None
    )

    assert [j.status for j in judgements] == [en.STATUS_ERROR]
    assert stats.jev_errors == 1


# ── Planning ─────────────────────────────────────────────────────────────────


def _judged(entity, winner, conf=0.9):
    names = en.candidate_names(entity["canonical_name"], entity["aliases"])
    return en.NameJudgement(entity, names, en.STATUS_RENAME, winner, conf)


def _planning_conn(existing=()):
    def responder(sql, params):
        if "WHERE canonical_name = ANY" in sql:
            return [_row(e) for e in existing if e["canonical_name"] in params[0]]
        return []

    return FakeConn(responder)


def _counts_session(counts):
    return FakeSession(
        lambda q, p: [{"entity_id": k, "mentions": v} for k, v in counts.items() if k in p["ids"]]
        if "UNWIND $ids" in q
        else []
    )


def test_plan_renames_in_place_when_no_other_row_has_the_name():
    target = _entity("hub", "Antropic", ["Anthropic", "Antropic AI", "Anthropic PBC"])
    plan = en.plan_actions(_planning_conn(), _counts_session({"hub": 500}), [_judged(target, "Anthropic")], PICK)

    (action,) = plan.actions
    assert (action.kind, action.survivor_id, action.dup_ids, action.winner) == (
        en.ACTION_RENAME, "hub", [], "Anthropic",
    )
    assert plan.blocked == []


def test_plan_merges_into_existing_exact_name_row_and_survivor_is_the_better_connected():
    typo_hub = _entity("typo", "Open AI", ["OpenAI", "Open-AI", "OpenAI Inc"])
    clean = _entity("clean", "OpenAI", ["OpenAI Inc."], age_days=30)
    session = _counts_session({"typo": 6826, "clean": 41})

    plan = en.plan_actions(_planning_conn([clean]), session, [_judged(typo_hub, "OpenAI")], PICK)

    (action,) = plan.actions
    assert action.kind == en.ACTION_MERGE
    assert action.survivor_id == "typo"  # more mentions: the hub keeps its id and edges
    assert action.dup_ids == ["clean"]
    assert "keep 'Open AI' (6826 mention(s)) as 'OpenAI'" in en.format_action(action)
    # exactly one mention-count query for the whole member set
    assert len(session.queries_matching("UNWIND $ids")) == 1


def test_plan_blocks_a_rename_into_a_name_held_by_an_incompatible_type():
    target = _entity("hub", "Mercury", ["Mercury Inc", "Mercury Corp", "Mercury Co"], etype="ORGANIZATION")
    planet = _entity("planet", "Mercury Corp", etype="LOCATION")

    plan = en.plan_actions(_planning_conn([planet]), _counts_session({}), [_judged(target, "Mercury Corp")], PICK)

    assert plan.actions == []
    (blocked,) = plan.blocked
    assert blocked.winner == "Mercury Corp" and "incompatible type" in blocked.reason


def test_plan_groups_two_renames_to_the_same_name_into_one_merge():
    """The 2026-10-19 failure: 'Antropic' and 'Boommy' were both corrected to
    'Anthropic' in one run, each planned as a rename in place, leaving two
    'Anthropic' rows the semantic merge never revisits."""
    hub = _entity("hub", "Antropic", ["Anthropic", "Antropic AI", "Anthropic PBC"])
    other = _entity("other", "Boommy", ["Anthropic", "Boom", "Boomy"])
    session = _counts_session({"hub": 4285, "other": 19})

    plan = en.plan_actions(
        _planning_conn(), session, [_judged(hub, "Anthropic", 0.96), _judged(other, "Anthropic", 0.91)], PICK
    )

    (action,) = plan.actions
    assert action.kind == en.ACTION_MERGE
    assert (action.survivor_id, action.dup_ids, action.winner) == ("hub", ["other"], "Anthropic")
    assert action.old_names == ["Antropic", "Boommy"]
    assert action.confidence == 0.96
    assert "merge 'Antropic' and 'Boommy' into 'Anthropic'" in en.format_action(action)
    assert plan.blocked == []


def test_plan_groups_same_name_renames_with_an_existing_row():
    hub = _entity("hub", "Antropic", ["Anthropic", "Antropic AI", "Anthropic PBC"])
    other = _entity("other", "Anthropik", ["Anthropic", "Anthropik AI", "Anthropik Inc"])
    clean = _entity("clean", "Anthropic", ["Anthropic PBC"], age_days=30)
    session = _counts_session({"hub": 4285, "other": 7, "clean": 32})

    plan = en.plan_actions(
        _planning_conn([clean]), session, [_judged(hub, "Anthropic"), _judged(other, "Anthropic")], PICK
    )

    (action,) = plan.actions
    assert action.survivor_id == "hub"
    assert sorted(action.dup_ids) == ["clean", "other"]
    # apply_actions re-checks members[0] against old_name: a target must come first
    assert action.members[0]["id"] == "hub" and action.old_name == "Antropic"


def test_plan_blocks_a_second_same_name_rename_with_an_incompatible_type():
    org = _entity("org", "Mercurey", ["Mercury", "Mercury Inc", "Mercury Co"], etype="ORGANIZATION")
    place = _entity("place", "Mercuri", ["Mercury", "Planet Mercury", "Mercurio"], etype="LOCATION")

    plan = en.plan_actions(
        _planning_conn(), _counts_session({}), [_judged(org, "Mercury"), _judged(place, "Mercury")], PICK
    )

    (action,) = plan.actions
    assert action.kind == en.ACTION_RENAME and action.survivor_id == "org"
    (blocked,) = plan.blocked
    assert blocked.entity["id"] == "place" and "in this run" in blocked.reason


def test_plan_ignores_non_rename_judgements():
    target = _entity("hub", "OpenAI", ["Open AI", "Open-AI", "OpenAI Inc"])
    kept = en.NameJudgement(target, ["OpenAI"], en.STATUS_KEEP, "OpenAI", 0.9)
    unsure = en.NameJudgement(target, ["OpenAI"], en.STATUS_UNCERTAIN)
    err = en.NameJudgement(target, ["OpenAI"], en.STATUS_ERROR, error="x")
    conn = _planning_conn()

    assert en.plan_actions(conn, _counts_session({}), [kept, unsure, err], PICK).actions == []


# ── Applying ─────────────────────────────────────────────────────────────────


def _apply_conn(entities):
    by_id = {e["id"]: e for e in entities}

    def responder(sql, params):
        if "WHERE id = ANY(%s::uuid[])" in sql:
            return [_row(by_id[i]) for i in params[0] if i in by_id]
        return []

    return FakeConn(responder)


def _merge_recorder(edges=7):
    calls = []

    def merge_fn(session, conn, survivor_id, dup_ids, members):
        calls.append((survivor_id, list(dup_ids), [m["id"] for m in members]))
        return edges

    return merge_fn, calls


def _action(members, survivor_id, winner, *, kind=None, confidence=0.9):
    return en.RenameAction(
        kind=kind or (en.ACTION_MERGE if len(members) > 1 else en.ACTION_RENAME),
        old_name=members[0]["canonical_name"],
        winner=winner,
        confidence=confidence,
        members=[dict(m, mentions=m.get("mentions", 0)) for m in members],
        survivor_id=survivor_id,
        dup_ids=[m["id"] for m in members if m["id"] != survivor_id],
    )


def test_apply_rename_in_place_moves_old_name_to_aliases_and_updates_both_stores():
    hub = _entity("hub", "Antropic", ["Anthropic", "Antropic AI", "Anthropic PBC"])
    conn, session = _apply_conn([hub]), FakeSession()
    merge_fn, merges = _merge_recorder()

    result = en.apply_actions(conn, session, [_action([hub], "hub", "Anthropic")], merge_fn, progress=lambda m: None)

    assert merges == []  # nothing to merge
    ((rename_cql, rename_params),) = session.calls
    assert "SET e.canonical_name = $name" in rename_cql
    assert rename_params == {"entity_id": "hub", "name": "Anthropic"}
    (sql, params), = conn.sql_matching("UPDATE entities SET canonical_name")
    # winner leaves the aliases, the OLD canonical name joins them
    assert params == ("Anthropic", ["Anthropic PBC", "Antropic", "Antropic AI"], "hub")
    assert conn.commits == 1
    assert (result.renamed, result.merged, result.touched_ids) == (1, 0, {"hub"})


def test_apply_merge_with_the_misspelled_hub_as_survivor_renames_it_afterwards():
    typo = _entity("typo", "Open AI", ["Open-AI", "OpenAI Inc", "Open A.I."])
    clean = _entity("clean", "OpenAI", ["OpenAI Corp"])
    conn, session = _apply_conn([typo, clean]), FakeSession()
    merge_fn, merges = _merge_recorder(edges=41)

    result = en.apply_actions(
        conn, session, [_action([typo, clean], "typo", "OpenAI")], merge_fn, progress=lambda m: None
    )

    assert merges == [("typo", ["clean"], ["typo", "clean"])]  # merge first ...
    (sql, params), = conn.sql_matching("UPDATE entities SET canonical_name")  # ... then rename
    assert params[0] == "OpenAI" and params[2] == "typo"
    assert params[1] == ["Open A.I.", "Open AI", "Open-AI", "OpenAI Corp", "OpenAI Inc"]
    assert "OpenAI" not in params[1]
    assert session.calls[0][1] == {"entity_id": "typo", "name": "OpenAI"}
    assert (result.renamed, result.merged, result.edges_repointed) == (1, 1, 41)
    assert result.touched_ids == {"typo", "clean"} and result.deleted_ids == {"clean"}


def test_apply_merge_where_the_existing_row_already_has_the_name_does_not_rename():
    typo = _entity("typo", "Open AI", ["Open-AI", "OpenAI Inc", "Open A.I."])
    clean = _entity("clean", "OpenAI", ["OpenAI Corp"])
    conn, session = _apply_conn([typo, clean]), FakeSession()
    merge_fn, merges = _merge_recorder()

    result = en.apply_actions(
        conn, session, [_action([typo, clean], "clean", "OpenAI")], merge_fn, progress=lambda m: None
    )

    assert merges == [("clean", ["typo"], ["typo", "clean"])]
    assert conn.sql_matching("UPDATE entities SET canonical_name") == []
    assert session.calls == []  # no Memgraph rename either
    assert (result.renamed, result.merged) == (0, 1)
    assert conn.commits == 1


def test_apply_skips_actions_that_overlap_an_earlier_one_in_the_same_run():
    a = _entity("a", "Googles", ["Google", "Google LLC", "Alphabet Google"])
    b = _entity("b", "Gooogle", ["Google", "Google Inc", "Googel"])
    clean = _entity("clean", "Google")
    conn, session = _apply_conn([a, b, clean]), FakeSession()
    merge_fn, merges = _merge_recorder()
    actions = [
        _action([a, clean], "a", "Google"),
        _action([b, clean], "b", "Google"),  # `clean` was consumed by the first action
    ]

    result = en.apply_actions(conn, session, actions, merge_fn, progress=lambda m: None)

    assert len(merges) == 1 and merges[0][0] == "a"
    assert result.skipped == 1
    assert result.touched_ids == {"a", "clean"}


def test_apply_skips_when_a_member_vanished_or_was_renamed_since_planning():
    hub = _entity("hub", "Antropic", ["Anthropic", "Antropic AI", "Anthropic PBC"])
    gone = _action([hub, _entity("gone", "Anthropic")], "hub", "Anthropic")
    conn = _apply_conn([hub])  # `gone` no longer exists
    merge_fn, merges = _merge_recorder()
    session = FakeSession()

    result = en.apply_actions(conn, session, [gone], merge_fn, progress=lambda m: None)
    assert (result.skipped, merges, session.calls, conn.commits) == (1, [], [], 0)

    renamed_meanwhile = _entity("hub", "Anthropic PBC", ["Anthropic", "Antropic AI", "Anthropic Inc"])
    stale = _action([hub], "hub", "Anthropic")
    result = en.apply_actions(_apply_conn([renamed_meanwhile]), session, [stale], merge_fn, progress=lambda m: None)
    assert result.skipped == 1 and session.calls == []


def test_apply_rolls_back_and_raises_when_a_write_fails():
    hub = _entity("hub", "Antropic", ["Anthropic", "Antropic AI", "Anthropic PBC"])
    conn = _apply_conn([hub])
    merge_fn, _ = _merge_recorder()
    session = FakeSession()
    session.run = MagicMock(side_effect=RuntimeError("memgraph down"))

    with pytest.raises(RuntimeError):
        en.apply_actions(conn, session, [_action([hub], "hub", "Anthropic")], merge_fn, progress=lambda m: None)

    assert conn.rollbacks == 1 and conn.commits == 0


# ── Orchestration ────────────────────────────────────────────────────────────


def _hub_conn(targets, existing=()):
    entities = {e["id"]: e for e in [*targets, *existing]}

    def responder(sql, params):
        if "cardinality(aliases) >=" in sql:
            assert params[0] == 3  # ENTITY_NAME_REVIEW_MIN_ALIASES
            return [_row(e) for e in targets][: params[1]]
        if "WHERE canonical_name = ANY" in sql:
            return [_row(e) for e in existing if e["canonical_name"] in params[0]]
        if "WHERE id = ANY(%s::uuid[])" in sql:
            return [_row(entities[i]) for i in params[0] if i in entities]
        return []

    return FakeConn(responder)


def test_fetch_name_targets_orders_hubs_by_alias_count_and_applies_limit():
    conn = FakeConn(lambda sql, params: [])
    en.fetch_name_targets(conn, 3, 500)
    (sql, params), = conn.executed
    assert "cardinality(aliases) >= %s" in sql
    assert "ORDER BY cardinality(aliases) DESC" in sql
    assert params == (3, 500)


def test_dry_run_prints_proposals_and_writes_nothing(monkeypatch):
    typo = _entity("typo", "Open AI", ["OpenAI", "Open-AI", "OpenAI Inc"])
    clean = _entity("clean", "OpenAI")
    conn = _hub_conn([typo], [clean])
    session = _counts_session({"typo": 100, "clean": 5})
    monkeypatch.setattr(jev, "system_one_many", _fake_many(lambda s, q: _answers(s["names"], 1)))
    merge_fn, merges = _merge_recorder()
    lines = []

    result = en.run_name_correction(
        conn, session, dry_run=True, limit=500, merge_fn=merge_fn, pick_survivor=PICK, progress=lines.append
    )

    assert conn.write_statements() == [] and conn.commits == 0
    assert session.write_queries() == [] and merges == []
    assert result.touched_ids == set()
    text = "\n".join(lines)
    assert "would merge 'Open AI' with existing 'OpenAI'" in text and "confidence 0.90" in text


def test_jev_errors_and_low_confidence_leave_everything_unchanged(monkeypatch):
    t1 = _entity("e1", "Open AI", ["OpenAI", "Open-AI", "OpenAI Inc"])
    t2 = _entity("e2", "Antropic", ["Anthropic", "Antropic AI", "Anthropic PBC"])
    conn = _hub_conn([t1, t2])
    session = _counts_session({})

    def per_state(state, questions):
        if state["names"][0] == "Open AI":
            return JevError("HTTP 500", status_code=500)
        return _answers(state["names"], 1, p=0.6)  # agrees, but below 0.8

    monkeypatch.setattr(jev, "system_one_many", _fake_many(per_state))
    merge_fn, merges = _merge_recorder()

    result = en.run_name_correction(
        conn, session, dry_run=False, limit=500, merge_fn=merge_fn, pick_survivor=PICK, progress=lambda m: None
    )

    assert merges == [] and conn.write_statements() == [] and conn.commits == 0
    assert session.write_queries() == []
    assert (result.renamed, result.merged, result.touched_ids) == (0, 0, set())


def test_execute_applies_the_confident_rename_and_reports_touched_ids(monkeypatch):
    hub = _entity("hub", "Antropic", ["Anthropic", "Antropic AI", "Anthropic PBC"])
    conn = _hub_conn([hub])
    session = _counts_session({"hub": 900})
    monkeypatch.setattr(jev, "system_one_many", _fake_many(lambda s, q: _answers(s["names"], 1)))
    merge_fn, _ = _merge_recorder()

    result = en.run_name_correction(
        conn, session, dry_run=False, limit=500, merge_fn=merge_fn, pick_survivor=PICK, progress=lambda m: None
    )

    assert result.renamed == 1 and result.touched_ids == {"hub"}
    assert conn.sql_matching("UPDATE entities SET canonical_name")
