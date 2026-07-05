---
title: Long-Running LLM Operations Need the Async Job Pattern, Not a Sync Endpoint With a Spinner
date: 2026-07-05
category: docs/solutions/architecture-patterns
module: community/theme-report generation
problem_type: architecture_pattern
component: background_job
severity: high
applies_when:
  - "an HTTP endpoint wraps a multi-step LLM-backed operation whose total duration is unbounded or highly variable"
  - "a sibling feature in the same codebase already has an established async job pattern (background thread + DB status column + client polling or SSE)"
  - "the operation fans out to multiple LLM calls (e.g. per-item analysis) followed by a final synthesis/aggregation call"
  - "failure-handling code catches broad exceptions and downgrades a status value instead of recording which step failed and why"
  - "users report a feature as 'stuck' or 'hung' when the backend is actually still processing"
symptoms:
  - "clicking 'Generate Report' shows a static 'Generating...' label with no progress signal and no way to tell in-progress from hung"
  - "backend has already completed and written the result to the DB but no in-flight request is visible in the logs at the moment a user asks if it's stuck"
  - "a report persists with status='completed' or 'partial' but core content fields (buckets, narrative) are empty with no recorded error"
  - "concurrency guard rows get stuck at their default 'running'/'generating' status until a stale-run reaper times them out"
root_cause: async_timing
resolution_type: code_fix
related_components:
  - src/rag/themes.py
  - src/rag/community_runs.py
  - src/rag/api/routes/themes.py
  - src/rag/api/routes/community.py
tags: [async-jobs, long-running-tasks, sse, exception-handling, background-thread, polling, llm-integration, silent-failure]
---

# Long-Running LLM Operations Need the Async Job Pattern, Not a Sync Endpoint With a Spinner

## Context

The research-workspace/communities feature was implemented by an external model ("DeepSeek") working directly against the repo, outside Claude Code, on branch `feat/research-workspace-communities` (plan: `docs/plans/2026-07-04-001-feat-research-workspace-communities-plan.md`, authored by "Fable"). The plan called for theme-report generation to move to a "just show progress" UX, but the shipped implementation kept `POST /api/themes` as a fully synchronous, blocking HTTP endpoint: it called `generate_theme_report()` directly and only returned once an entire multi-community LLM fan-out (via `ThreadPoolExecutor`) plus a final cross-community synthesis call had completed. The frontend (`CommunityView.tsx`) did a single blocking `await` behind a button whose label was a static string, `"Generating..."` — no timeout, no progress signal, no way to tell "still working" from "hung."

The codebase already had the right answer sitting right next to it. The sibling community-detection-run feature (`src/rag/community_runs.py`) had already solved exactly this shape of problem: a `threading.Thread` background worker, a DB status column, and SSE-based polling (`stream_run_events`) so the client always has a live answer to "is this still going?" Theme-report generation needed the same shape and didn't get it — a classic case of a new feature reinventing (badly) a solved problem instead of reusing the established pattern.

This stopped being theoretical mid-session: the user clicked "Generate Report," the button sat on "Generating..." indefinitely, and asked "is it progressing or stuck? what is going on with this run?" Investigation found the report had actually completed successfully in the DB — but there was no way to see that from outside, because the fully-synchronous architecture has no intermediate visible state at all: the DB row only gets written once the whole operation is done.

Later, after the async fix below had shipped, the same "is it hung or is it just a lot of work?" question came back for real: a 43-community run took 13.5 minutes end-to-end. This time the async+polling UX correctly showed "in progress" the whole time, and the live debugging techniques in this doc were what separated "definitely still working" from "maybe dead."

## Guidance

### 1. Split the operation into a fast, non-blocking entry point and a reusable "do the work" function

The core pattern, as implemented in `src/rag/themes.py`:

- **`_create_report_row(run_id, model, placeholder_status)`** — validates the run exists, inserts a placeholder `theme_reports` row with a caller-supplied status, and returns `(report_id, communities)`. This is the only part that needs to happen before the caller gets a response.
- **`_run_analysis(report_id, communities, model)`** — does all the actual heavy lifting: the per-community LLM fan-out (`ThreadPoolExecutor`), the cross-community synthesis call, and the final `UPDATE` that writes the terminal status and result. This function doesn't know or care whether it's running inline or in a background thread.
- **`generate_theme_report(run_id, model=None)`** — the original synchronous entry point, kept for callers that are fine blocking (the CLI, the MCP server): inserts the placeholder row with `placeholder_status="completed"`, then blocks on `_run_analysis` before returning. This still uses the terminal-sounding `"completed"` placeholder flagged as a smell in point 2 below — it's tolerable here only because nothing can observe the row between insert and the caller's blocking return, so no other reader is ever exposed to the lie. Don't copy this specific choice into a new sync entry point without that same guarantee.
- **`start_theme_report(run_id, model=None)`** — the new async entry point used only by the REST route: inserts the placeholder row with `placeholder_status="generating"`, spawns `_run_analysis` in a `threading.Thread(daemon=True)`, and returns the `report_id` immediately, well before the work is done.

```python
def _create_report_row(run_id: str, model: str, placeholder_status: str) -> tuple[str, list[dict]]:
    # ...validate run, look up communities...
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
    # per-community LLM fan-out via ThreadPoolExecutor, then cross-community
    # synthesis call, then a single terminal UPDATE writing status + result.
    ...


def generate_theme_report(run_id: str, model: str | None = None) -> str:
    """Generate a theme report synchronously, blocking until complete. Used by the CLI and MCP server."""
    report_id, communities = _create_report_row(run_id, model, placeholder_status="completed")
    _run_analysis(report_id, communities, model)
    return report_id


def start_theme_report(run_id: str, model: str | None = None) -> str:
    """Create a theme report and run analysis in a background thread, returning immediately."""
    report_id, communities = _create_report_row(run_id, model, placeholder_status="generating")
    thread = threading.Thread(target=_run_analysis, args=(report_id, communities, model), daemon=True)
    thread.start()
    return report_id
```

The same split was applied to `regenerate_report()` / `start_regenerate_report()`. The REST route (`src/rag/api/routes/themes.py`) calls the `start_*` variants and returns `{"id": ..., "status": "generating"}` immediately; the CLI and MCP server keep calling the blocking `generate_theme_report()` / `regenerate_report()` because they're fine waiting and don't need a polling contract.

The generalizable shape: **one function that inserts a row and returns fast, one function that does the real work and is agnostic about who calls it, and two thin wrappers — one that awaits inline, one that backgrounds it.** Don't write two divergent implementations of the actual analysis logic for the sync and async paths.

### 2. The migration needed to support an honest in-progress status

Before this fix, the placeholder row was always inserted with `status='completed'` — a status that was simply a lie until the row got overwritten. Supporting a real in-progress state requires widening the CHECK constraint and changing the column default (see `scripts/migrate/014_theme_reports_async.sql`):

```sql
-- Theme report generation was previously a fully synchronous HTTP request;
-- the row was inserted with a 'completed' placeholder status and immediately
-- overwritten once analysis finished. Generation now runs in a background
-- thread so the API can return immediately and the frontend can poll for
-- progress, so the placeholder status needs to be a distinct, real state.

ALTER TABLE theme_reports DROP CONSTRAINT IF EXISTS theme_reports_status_check;
ALTER TABLE theme_reports ADD CONSTRAINT theme_reports_status_check
  CHECK (status IN ('generating', 'completed', 'partial', 'failed'));
ALTER TABLE theme_reports ALTER COLUMN status SET DEFAULT 'generating';
```

If a placeholder status ever defaults to a terminal-sounding value (`'completed'`, `'done'`, `'success'`) before the work has actually started, treat that as a smell: it means "in progress" was never a first-class state in the schema, which is exactly the gap that made the original hang report indistinguishable from success.

### 3. The client-side contract: poll, show elapsed time, gate the result behind a terminal status

A static `"Generating..."` label gives the user zero information about whether the operation is alive. `CommunityView.tsx`'s `handleGenerateTheme` instead: kicks off generation, starts a 1-second ticking counter for the button label, and polls the status endpoint every 1.5 seconds until a terminal status is reached — only then does it reveal the "View Theme Report" link (or "View Failed Report" if it failed).

```tsx
async function pollThemeStatus(reportId: string) {
  while (mountedRef.current) {
    await new Promise((resolve) => window.setTimeout(resolve, 1500));
    if (!mountedRef.current) return;
    try {
      const report = await getTheme(reportId);
      if (!mountedRef.current) return;
      if (report.status !== "generating") {
        setThemeStatus(report.status);
        return;
      }
    } catch (e) {
      if (mountedRef.current) setError(e instanceof Error ? e.message : "Failed to check theme report status");
      return;
    }
  }
}

async function handleGenerateTheme() {
  if (!runId || themeGenerating) return;
  setThemeGenerating(true);
  setThemeElapsedSec(0);
  const tick = window.setInterval(() => setThemeElapsedSec((s) => s + 1), 1000);
  try {
    const resp = await generateTheme(runId);
    setThemeGeneratedId(resp.id);
    setThemeStatus("generating");
    await pollThemeStatus(resp.id);
  } finally {
    window.clearInterval(tick);
    if (mountedRef.current) setThemeGenerating(false);
  }
}
```

The button label becomes `Generating… (47s)` instead of a frozen string, so a genuinely slow-but-working run visibly ticks up seconds instead of looking dead. `LibraryView.tsx` applies the same pattern when a user opens a report that's still `status='generating'`: it auto-polls and shows "Generating report… this page updates automatically." The result view/link is only ever revealed once the poll returns `completed`, `partial`, or `failed` — never while still `generating` — so there's no window where a half-written result could be shown as done.

### 4. Never let a broad `except Exception:` around a sub-step silently discard the real error while downgrading status

This is the reinforcing lesson, and it's not hypothetical — it happened in this exact code path. `_run_analysis`'s cross-community synthesis step is wrapped like this:

```python
if ordered:
    try:
        synthesis_prompt = prompts.THEME_SYNTHESIS.format(analyses=json.dumps(ordered))
        raw = _call_llm(synthesis_prompt, model)
        synthesis = _parse_json(raw)
    except Exception:
        synthesis = {"buckets": [], "narrative": "", "cleanup_recommendations": []}
        if status == "completed":
            status = "partial"
```

On the live 43-community run, 34 communities succeeded and 9 failed (expected — those get recorded in `failed_community_ids`). But the synthesis call *also* failed, and this `except Exception:` block caught it, silently substituted empty buckets/narrative/cleanup_recommendations, and left `status='partial'` — a status that's indistinguishable from "some individual communities failed, but synthesis worked fine." Nothing in the persisted row records that synthesis itself was the thing that failed, or why. A later reader has no way to tell "9 communities didn't have enough data to analyze" apart from "the synthesis prompt timed out on 34 combined analyses" — they look identical from the DB, and the only way to find out is to reverse-engineer it from the fact that all three synthesis fields are empty.

The fix pattern for any similar sub-step: at minimum, log the actual exception instead of swallowing it silently. Better: record which specific step failed and why, so a human (or the next debugging session) doesn't have to infer causation from absence.

```python
# AFTER — logs the real cause and records which step failed, instead of guessing later
except Exception as e:
    logger.exception("theme synthesis failed for report %s", report_id)
    synthesis = {"buckets": [], "narrative": "", "cleanup_recommendations": []}
    synthesis_error = str(e)
    if status == "completed":
        status = "partial"
```

### 5. A reusable debugging technique: is the background job actually stuck, or just slow?

When a background thread's progress isn't otherwise observable, and you're inside a slim container with no `netstat`/`ss`, you can still get direct evidence of in-flight work by reading `/proc/net/tcp` from inside the worker process's container:

- Look for a connection in TCP state `01` (ESTABLISHED) to the expected remote host on port `443` (hex `01BB`) — this is a live, in-flight outbound HTTPS request. In this session it confirmed an active call to the LLM provider (`opencode.ai`/`openrouter.ai`) at the exact moment the user asked "is it hung?"
- Cross-check the observed duration against known concurrency/timeout math. Here: 43 communities, `COMMUNITY_SUMMARY_MAX_WORKERS=4` → ~11 sequential rounds through the thread pool, each call capped at a 120s `requests.post(..., timeout=120)` — yielding an expected range of roughly 10-20+ minutes against a slower model (`deepseek-v4-pro`). The observed total (13.5 minutes) fell squarely inside that bound, which is strong evidence the process was working, not deadlocked — a hard per-call timeout means the process literally cannot hang indefinitely; it can only be "slow across many sequential calls."

This generalizes to any "is my background job stuck?" question in this stack: check for an ESTABLISHED outbound connection as proof of activity, and validate the elapsed time against the known concurrency/timeout ceiling before assuming something is broken.

## Why This Matters

A synchronous HTTP endpoint wrapping a long-running, multi-step LLM operation is architecturally indistinguishable from a hang — even when it's working perfectly. The DB shows success eventually, but there is no visible "in progress" state anywhere along the way: no request appears to be happening, no status ticks forward, nothing distinguishes "still computing" from "crashed silently." This erodes user trust (the exact "is it hung or is it just a lot of work?" question this session had to answer twice) and it makes debugging needlessly hard, since the only tool available is guessing.

Combined with silent exception-swallowing, the failure mode compounds: not only can the user not tell if something is happening, but when something *does* go wrong, the system actively hides which part failed. The synthesis-step bug in this session is the clearest possible illustration — the async-job fix correctly solved "is this still running," but a broad `except Exception:` two lines away made "did this actually work, and if not, why" just as opaque as the original hang problem was for progress visibility.

The meta-lesson: when a codebase already has an established pattern for a class of problem — here, `community_runs.py`'s thread + status-column + polling pattern for "long-running background work with progress visibility" — a new feature with the same shape should be built on that pattern from the start. Bolting on a synchronous implementation and patching it later with client-side spinners or timeouts is strictly worse than reusing the existing pattern up front: it costs an extra debugging cycle, an extra migration, and an extra round of user-facing confusion that the sibling feature had already paid for and solved.

## When to Apply

Apply the async job pattern (background thread/worker + DB status column + client polling) whenever a new feature:

- Calls an LLM or other slow external API from inside a synchronous request handler.
- Does multi-step fan-out work (e.g., a `ThreadPoolExecutor` across N items, or any per-item + aggregate-synthesis two-phase job) whose total duration scales with N and isn't bounded to sub-second/low-second latency.
- Has a sibling feature in the same codebase that already solves this "long-running + progress visibility" shape. In this codebase, `src/rag/community_runs.py` is the reference implementation — copy its thread/status/polling shape for any future long-running feature rather than re-deriving it.

Apply the "don't swallow sub-step exceptions" guidance whenever a multi-step pipeline has an aggregate/synthesis/finalization step that can fail independently of the per-item steps that feed it — record (at minimum log, ideally persist) which specific step failed, not just that "something in the pipeline" caused a status downgrade.

## Examples

**(a) Route: synchronous before vs. async start/poll after**

```python
# BEFORE — src/rag/api/routes/themes.py, blocking the whole request
@router.post("/api/themes")
def create_theme_report(req: ThemeRequest):
    report_id = generate_theme_report(req.run_id, req.model)  # blocks for minutes
    return {"id": report_id, "status": "completed"}

# AFTER — returns immediately, work happens in a background thread
@router.post("/api/themes")
def create_theme_report(req: ThemeRequest):
    report_id = start_theme_report(req.run_id, req.model)
    return {"id": report_id, "status": "generating"}
```

**(b) Status CHECK constraint before vs. after**

```sql
-- BEFORE (implicit): status defaulted to 'completed' even for a placeholder row
-- that had not been analyzed yet — the schema had no real "in progress" state.

-- AFTER (scripts/migrate/014_theme_reports_async.sql):
ALTER TABLE theme_reports DROP CONSTRAINT IF EXISTS theme_reports_status_check;
ALTER TABLE theme_reports ADD CONSTRAINT theme_reports_status_check
  CHECK (status IN ('generating', 'completed', 'partial', 'failed'));
ALTER TABLE theme_reports ALTER COLUMN status SET DEFAULT 'generating';
```

**(c) Frontend before vs. after**

```tsx
// BEFORE — a single blocking await behind a static label
<button onClick={async () => {
  setGenerating(true);
  await generateTheme(runId);   // caller has no idea how long this will take
  setGenerating(false);
}}>
  {generating ? "Generating..." : "Generate Theme Report"}
</button>

// AFTER — kick off, poll, show elapsed time, gate the result on terminal status
<button onClick={handleGenerateTheme} disabled={themeGenerating}>
  {themeGenerating ? `Generating… (${themeElapsedSec}s)` : "Generate Theme Report"}
</button>
{themeGeneratedId && !themeGenerating ? (
  <a href={`/reports/${themeGeneratedId}`}>
    {themeStatus === "failed" ? "View Failed Report" : "View Theme Report"}
  </a>
) : null}
```

## Related, Not Yet Fixed

The cross-community synthesis silent-failure bug described in guidance point 4 is still open as of this writing. On the real 43-community run that motivated this write-up, 34/43 communities succeeded, but `report.buckets`, `report.narrative`, and `report.cleanup_recommendations` all came back empty with no persisted reason — the synthesis call failed inside a bare `except Exception:` that only downgraded `status` to `'partial'` without recording what actually broke (suspected: a timeout or malformed JSON response when asked to synthesize 34 communities' worth of analysis in a single combined prompt, but this is unconfirmed). This is exactly the evidence that proved the "never swallow the real exception" guidance above, and is natural follow-up work: at minimum, log the exception; ideally, persist a `synthesis_error` field distinguishing "synthesis never ran," "synthesis timed out," and "synthesis returned unparseable JSON" from each other.

## Adjacent, Fully-Closed Bug: SSE separator mismatch

A sharp, narrow bug hit in the same session while wiring up community-run progress events over Server-Sent Events, worth flagging because it's easy to reintroduce in any new SSE endpoint in this codebase. The backend used `sse-starlette`'s `EventSourceResponse` with its *default* separator, which is `"\r\n"` (CRLF). The frontend's hand-rolled SSE parser (mirroring the pattern used by `/api/answer/stream`) splits events on bare `"\n\n"` (LF), so it never found a boundary to split on — community-run progress and results silently never appeared in the UI, even though the backend was emitting events correctly.

Root-caused precisely by importing `sse_starlette.sse.ensure_bytes` directly and inspecting the actual byte output, confirming `\r\n\r\n` framing. Fixed with a one-line change in `src/rag/api/routes/community.py`:

```python
# Before: EventSourceResponse(event_iter())          — defaults to "\r\n" separator
# After:
return EventSourceResponse(event_iter(), sep="\n")
```

This fix was verified in both directions in the same session: reverted with `sed`, re-ran a new regression test (`tests/test_api_community.py::test_run_events_uses_lf_separator`), confirmed it failed against the literal `\r\n` bytes in the response body; restored the fix, confirmed the test passed. The frontend parser was also hardened to tolerate either separator via regex as defense in depth. Any future SSE endpoint added to this codebase should either explicitly pass `sep="\n"` to `EventSourceResponse` to match the existing hand-rolled parsers, or update the parser to match whatever separator the backend actually emits — don't assume the library default matches what's already deployed elsewhere in the app.

## Related

- Plan: `docs/plans/2026-07-04-001-feat-research-workspace-communities-plan.md` (authored by "Fable"; implemented by an external model, "DeepSeek," outside Claude Code)
- Reference implementation to copy for future long-running features: `src/rag/community_runs.py`
- No existing `docs/solutions/` entries overlap with this topic (checked `architecture-patterns/`, `database-issues/`, `performance-issues/` — all cover unrelated Docker/Postgres/retrieval-performance problems)
