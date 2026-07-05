from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
import requests

from rag import prompts
from rag.config import settings
from rag.db import get_connection


def _call_llm(prompt_text: str, model: str) -> str:
    api_key = settings.OPENCODE_API_KEY or settings.OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError("No LLM API key configured (set OPENCODE_API_KEY or OPENROUTER_API_KEY)")
    if settings.OPENCODE_API_KEY:
        url = "https://opencode.ai/zen/go/v1/chat/completions"
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _create_report_row(run_id: str, model: str, placeholder_status: str) -> tuple[str, list[dict]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, status, result FROM community_runs WHERE id = %s",
            (run_id,),
        ).fetchone()

    if row is None or row[1] != "completed":
        raise ValueError("Run not found or not completed")

    result_data = row[2]
    if isinstance(result_data, str):
        result_data = json.loads(result_data)
    communities = result_data.get("communities", []) if isinstance(result_data, dict) else []

    if not communities:
        raise ValueError("Run has no communities")

    with get_connection() as conn:
        report_row = conn.execute(
            """INSERT INTO theme_reports (run_id, status, model)
               VALUES (%s, %s, %s)
               RETURNING id""",
            (run_id, placeholder_status, model),
        ).fetchone()
        conn.commit()
    return str(report_row[0]), communities


def _run_analysis(report_id: str, communities: list[dict], model: str) -> None:
    prompt_override = settings.THEME_REPORT_PROMPT
    failed_ids: list[int] = []
    max_workers = settings.COMMUNITY_SUMMARY_MAX_WORKERS

    def analyze_one(i: int, community: dict):
        try:
            entities_json = json.dumps([e.get("canonical_name", "") for e in community.get("entities", [])])
            sources_json = json.dumps([s.get("source_name", "") for s in community.get("contributing_sources", [])])
            chunks = [{"content": c.get("content", ""), "source": c.get("source_name", "")} for c in community.get("chunks", [])]
            chunks_json = json.dumps(chunks)
            is_cross = community.get("is_cross_source", False)

            analysis_prompt = prompt_override or prompts.THEME_COMMUNITY_ANALYSIS
            text = analysis_prompt.format(
                entities=entities_json,
                sources=sources_json,
                chunks=chunks_json,
                cross_source=str(is_cross),
            )
            raw = _call_llm(text, model)
            result = _parse_json(raw)
            result["evidence_chunks"] = chunks
            result["all_entities"] = [e.get("canonical_name", "") for e in community.get("entities", [])]
            return i, result
        except Exception:
            raise

    successful: dict[int, dict] = {}

    if max_workers > 1 and len(communities) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(analyze_one, i, c): i
                for i, c in enumerate(communities)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    i, analysis = future.result()
                    successful[i] = analysis
                except Exception:
                    failed_ids.append(idx)
    else:
        for i, community in enumerate(communities):
            try:
                _, analysis = analyze_one(i, community)
                successful[i] = analysis
            except Exception:
                failed_ids.append(i)

    if failed_ids and not successful:
        status = "failed"
    elif failed_ids:
        status = "partial"
    else:
        status = "completed"

    ordered = [successful[i] for i in sorted(successful)]

    if ordered:
        try:
            synthesis_prompt = prompts.THEME_SYNTHESIS.format(
                analyses=json.dumps(ordered),
            )
            raw = _call_llm(synthesis_prompt, model)
            synthesis = _parse_json(raw)
        except Exception:
            synthesis = {"buckets": [], "narrative": "", "cleanup_recommendations": []}
            if status == "completed":
                status = "partial"
    else:
        synthesis = {"buckets": [], "narrative": "", "cleanup_recommendations": []}

    report = {
        "communities": ordered,
        "buckets": synthesis.get("buckets", []),
        "narrative": synthesis.get("narrative", ""),
        "cleanup_recommendations": synthesis.get("cleanup_recommendations", []),
    }

    with get_connection() as conn:
        conn.execute(
            """UPDATE theme_reports
               SET status = %s,
                   failed_community_ids = %s::jsonb,
                   report = %s::jsonb
               WHERE id = %s""",
            (status, json.dumps(failed_ids), json.dumps(report), report_id),
        )
        conn.commit()


def generate_theme_report(run_id: str, model: str | None = None) -> str:
    """Generate a theme report synchronously, blocking until complete. Used by the CLI and MCP server."""
    model = model or settings.THEME_REPORT_MODEL or "deepseek-v4-pro"
    report_id, communities = _create_report_row(run_id, model, placeholder_status="completed")
    _run_analysis(report_id, communities, model)
    return report_id


def start_theme_report(run_id: str, model: str | None = None) -> str:
    """Create a theme report and run analysis in a background thread, returning immediately."""
    model = model or settings.THEME_REPORT_MODEL or "deepseek-v4-pro"
    report_id, communities = _create_report_row(run_id, model, placeholder_status="generating")
    thread = threading.Thread(target=_run_analysis, args=(report_id, communities, model), daemon=True)
    thread.start()
    return report_id


def get_report(report_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            """SELECT id, run_id, status, failed_community_ids, report,
                      model, created_at
               FROM theme_reports WHERE id = %s""",
            (report_id,),
        ).fetchone()
    if row is None:
        return None
    return _report_row_to_dict(row)


def list_reports(limit: int = 20, offset: int = 0) -> dict:
    with get_connection() as conn:
        total = conn.execute("SELECT count(*) FROM theme_reports").fetchone()[0]
        rows = conn.execute(
            """SELECT id, run_id, status, failed_community_ids, report,
                      model, created_at
               FROM theme_reports
               ORDER BY created_at DESC LIMIT %s OFFSET %s""",
            (limit, offset),
        ).fetchall()
    return {
        "reports": [_report_row_to_dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def _resolve_regenerate(report_id: str, model: str | None) -> tuple[str, str]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT run_id, failed_community_ids, model FROM theme_reports WHERE id = %s",
            (report_id,),
        ).fetchone()
    if row is None:
        raise ValueError("Report not found")
    run_id = str(row[0])
    model = model or row[2] or settings.THEME_REPORT_MODEL or "deepseek-v4-pro"
    return run_id, model


def regenerate_report(report_id: str, model: str | None = None) -> str:
    """Regenerate a theme report synchronously, blocking until complete. Used by the CLI and MCP server."""
    run_id, resolved_model = _resolve_regenerate(report_id, model)
    return generate_theme_report(run_id, resolved_model)


def start_regenerate_report(report_id: str, model: str | None = None) -> str:
    """Regenerate a theme report in a background thread, returning immediately."""
    run_id, resolved_model = _resolve_regenerate(report_id, model)
    return start_theme_report(run_id, resolved_model)


def _report_row_to_dict(row) -> dict:
    report = row[4]
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except (json.JSONDecodeError, TypeError):
            report = {}
    failed = row[3]
    if isinstance(failed, str):
        try:
            failed = json.loads(failed)
        except (json.JSONDecodeError, TypeError):
            failed = []
    return {
        "id": str(row[0]),
        "run_id": str(row[1]),
        "status": row[2],
        "failed_community_ids": failed if isinstance(failed, list) else [],
        "report": report if isinstance(report, dict) else {},
        "model": row[5] or "",
        "created_at": row[6].isoformat() if row[6] else None,
    }
