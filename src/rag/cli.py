import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.config import settings


# Heavy modules are imported lazily so API-mode CLI commands — which dispatch
# through RagClient and never touch these — don't pay their import cost on every
# invocation. rag.graph_db pulls in neo4j+pandas (~0.45s) and rag.retrieval the
# embedding/graph stack (~0.2s); deferring both cuts CLI startup from ~0.75s to
# ~0.1s. Kept as module-level names so tests can patch rag.cli.<name>, mirroring
# the submit_ingestion_job wrapper below.
def get_connection(*args, **kwargs):
    from rag.db import get_connection as _fn

    return _fn(*args, **kwargs)


def get_graph_driver(*args, **kwargs):
    from rag.graph_db import get_graph_driver as _fn

    return _fn(*args, **kwargs)


def hybrid_search(*args, **kwargs):
    from rag.retrieval import hybrid_search as _fn

    return _fn(*args, **kwargs)


def retrieve(*args, **kwargs):
    from rag.retrieval import retrieve as _fn

    return _fn(*args, **kwargs)

# Binary documents are converted to self-contained markdown on the CLI (rag.prepare)
# before a job is queued; text formats are submitted as-is. The backend worker
# parses markdown/text only.
BINARY_EXTENSIONS = {".pdf", ".docx", ".pptx"}
TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
SUPPORTED_EXTENSIONS = BINARY_EXTENSIONS | TEXT_EXTENSIONS
STAGE_ORDER = (
    "parsing",
    "profiling",
    "chunking",
    "validation",
    "embedding",
    "graph_extraction",
    "graph_linking",
    "insight_extraction",
)

app = typer.Typer(help="RAG CLI — document ingestion and management")
sources_app = typer.Typer(help="Manage ingested sources")
jobs_app = typer.Typer(help="Manage ingestion jobs")
community_app = typer.Typer(help="Community summarization")
worker_app = typer.Typer(help="Manage background workers (server-side)")
app.add_typer(sources_app, name="sources")
app.add_typer(jobs_app, name="jobs")
app.add_typer(community_app, name="community")
app.add_typer(worker_app, name="worker")


def _use_api() -> bool:
    """Return True when the CLI should route commands through the API."""
    from rag.cli_config import load_cli_config

    cfg = load_cli_config()
    return bool(cfg.server_url and cfg.api_key)


def _get_client():
    """Build a RagClient using config (env vars > config file)."""
    from rag.api_client import RagClient

    return RagClient.from_config()

console = Console()


def _parse_key_value_pairs(items: Optional[list[str]], label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid {label} format: {item!r} (expected key=value)")
        key, _, value = item.partition("=")
        parsed[key.strip()] = value.strip()
    return parsed


def _get_connection():
    return get_connection()


def _get_graph_driver():
    return get_graph_driver()


def submit_ingestion_job(*args, **kwargs):
    from rag.ingestion import submit_ingestion_job as _fn
    return _fn(*args, **kwargs)


# Lazy wrappers for the heavy prepare path (Docling) and image description, kept
# as module-level names so tests can patch rag.cli.<name> without importing
# Docling. Only invoked for binary documents.
def prepare_binary(*args, **kwargs):
    from rag.prepare import prepare_binary as _fn
    return _fn(*args, **kwargs)


def finalize_markdown(*args, **kwargs):
    from rag.prepare import finalize_markdown as _fn
    return _fn(*args, **kwargs)


def describe_image(*args, **kwargs):
    from rag.image_description import describe_image as _fn
    return _fn(*args, **kwargs)


def retry_job(*args, **kwargs):
    from rag.ingestion import retry_job as _fn
    return _fn(*args, **kwargs)


def cancel_job(*args, **kwargs):
    from rag.ingestion import cancel_job as _fn
    return _fn(*args, **kwargs)


def detect_communities(*args, **kwargs):
    from rag.community import detect_communities as _detect_communities

    return _detect_communities(*args, **kwargs)


@app.command()
def health() -> None:
    """Check Postgres and Memgraph connectivity."""
    if _use_api():
        try:
            with _get_client() as client:
                client.health()
        except Exception as e:
            console.print(f"[red]Unhealthy: {e}[/red]")
            raise typer.Exit(1)
        console.print("[green]Ready: server is reachable.[/green]")
        return
    try:
        with _get_connection() as conn:
            conn.execute("SELECT 1").fetchone()

        with _get_graph_driver() as driver:
            with driver.session() as session:
                session.run("RETURN 1")
    except Exception as e:
        console.print(f"[red]Unhealthy: {e}[/red]")
        raise typer.Exit(1)

    console.print("[green]Ready: Postgres and Memgraph are reachable.[/green]")


def _prepare_markdown(file: Path, describe):
    """Convert a binary document to self-contained markdown, describing images.

    ``describe`` turns image bytes into text (backend API in API mode, local call
    in direct-DB mode). Raises RuntimeError with a "Preparation failed" prefix so
    the caller reports a local-preparation error distinctly from a backend
    submission error, and never queues a job when preparation fails.
    """
    from rag.prepare import PrepareError

    try:
        prepared = prepare_binary(file)
        content = finalize_markdown(prepared, describe)
    except PrepareError as exc:
        raise RuntimeError(f"Preparation failed: {exc}") from exc
    return content, prepared


def _ingest_one_api(client, file: Path, name: Optional[str], metadata: dict):
    """Ingest a single file in API mode: prepare binaries locally, submit markdown."""
    if file.suffix.lower() in BINARY_EXTENSIONS:
        content, prepared = _prepare_markdown(
            file, lambda data, mime: client.describe_image(data, mime)
        )
        meta = {**metadata, "prepared_image_count": prepared.image_count}
        return client.submit_text(
            content,
            name=name,
            metadata=meta,
            original_md5=prepared.original_md5,
            file_name=prepared.original_filename,
            file_type=prepared.original_extension,
        )
    return client.submit_ingest(file, name=name, metadata=metadata)


def _ingest_one_direct(file: Path, name: Optional[str], metadata: dict):
    """Ingest a single file in direct-DB mode: prepare binaries with a local caption."""
    if file.suffix.lower() in BINARY_EXTENSIONS:
        content, prepared = _prepare_markdown(
            file, lambda data, mime: describe_image(data, mime)
        )
        import tempfile

        tmp = tempfile.NamedTemporaryFile(
            prefix="prepared-", suffix=".md", delete=False, mode="w", encoding="utf-8"
        )
        tmp_path = Path(tmp.name)
        try:
            tmp.write(content)
            tmp.close()
            meta = {**metadata, "prepared_image_count": prepared.image_count}
            return submit_ingestion_job(
                tmp_path,
                name=name,
                metadata=meta,
                original_md5=prepared.original_md5,
                original_file_name=prepared.original_filename,
                original_file_type=prepared.original_extension,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    return submit_ingestion_job(file, name=name, metadata=metadata)


@app.command()
def ingest(
    paths: Annotated[list[Path], typer.Argument(help="Files or a single folder to ingest")],
    name: Annotated[Optional[str], typer.Option(help="Display name (single file only)")] = None,
    metadata: Annotated[
        Optional[list[str]],
        typer.Option("--metadata", "-m", help="Metadata as key=value pairs"),
    ] = None,
) -> None:
    """Ingest one or more documents, or all supported files in a folder."""
    try:
        parsed_metadata = _parse_key_value_pairs(metadata, "metadata")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Resolve file list
    if len(paths) == 1 and paths[0].is_dir():
        folder = paths[0]
        resolved_files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not resolved_files:
            console.print(f"[yellow]No supported files found in {folder}[/yellow]")
            raise typer.Exit(0)
        if name:
            console.print("[yellow]--name is ignored when ingesting a folder[/yellow]")
        name = None
    else:
        resolved_files = list(paths)
        if len(resolved_files) > 1 and name:
            console.print("[yellow]--name is ignored when ingesting multiple files[/yellow]")
            name = None

    # Submit each file
    results = []
    errors = []
    if _use_api():
        with _get_client() as client:
            for file in resolved_files:
                per_file_name = name if len(resolved_files) == 1 else None
                try:
                    result = _ingest_one_api(client, file, per_file_name, parsed_metadata)
                    results.append((file.name, result["job_id"], result["status"]))
                except FileNotFoundError as e:
                    errors.append((file.name, str(e)))
                except Exception as e:
                    errors.append((file.name, str(e)))
    else:
        for file in resolved_files:
            per_file_name = name if len(resolved_files) == 1 else None
            try:
                result = _ingest_one_direct(file, per_file_name, parsed_metadata)
                results.append((file.name, result["job_id"], result["status"]))
            except FileNotFoundError as e:
                errors.append((file.name, str(e)))
            except Exception as e:
                errors.append((file.name, str(e)))

    # Output table
    table = Table(title="Submitted Jobs")
    table.add_column("File", style="bold")
    table.add_column("Job ID", style="dim", no_wrap=True)
    table.add_column("Status")
    for file_name, job_id, status in results:
        table.add_row(file_name, job_id, "[cyan]pending[/cyan]")
    for file_name, error_msg in errors:
        table.add_row(file_name, "-", f"[red]{error_msg}[/red]")
    console.print(table)

    if results:
        console.print("[dim]Run [bold]rag worker[/bold] to process queued jobs.[/dim]")
    if errors and not results:
        raise typer.Exit(1)


@worker_app.command("launch")
def worker_launch(
    n: Annotated[int, typer.Argument(help="Number of workers to launch")] = 1,
) -> None:
    """Launch N background workers on the server."""
    with _get_client() as client:
        try:
            result = client.launch_workers(n)
        except Exception as e:
            console.print(f"[red]Failed to launch workers: {e}[/red]")
            raise typer.Exit(1)
    for wid in result.get("ids", []):
        console.print(wid)


@worker_app.command("stop")
def worker_stop(
    worker_id: Annotated[Optional[str], typer.Argument(help="Worker ID to stop")] = None,
    all_workers: Annotated[bool, typer.Option("--all", help="Stop every active worker")] = False,
) -> None:
    """Stop a background worker (or every active worker with --all)."""
    if all_workers and worker_id:
        console.print("[red]Pass either a worker ID or --all, not both.[/red]")
        raise typer.Exit(2)
    if not all_workers and not worker_id:
        console.print("[red]Provide a worker ID or use --all.[/red]")
        raise typer.Exit(2)

    with _get_client() as client:
        try:
            if all_workers:
                result = client.stop_all_workers()
            else:
                client.stop_worker(worker_id)
        except Exception as e:
            console.print(f"[red]Failed to stop worker: {e}[/red]")
            raise typer.Exit(1)

    if all_workers:
        stopped = result.get("stopped", [])
        if not stopped:
            console.print("[dim]No active workers.[/dim]")
        else:
            for wid in stopped:
                console.print(f"[green]Stopped {wid}[/green]")
    else:
        console.print(f"[green]Worker {worker_id} stopped.[/green]")


@worker_app.command("list")
def worker_list(
    show_all: Annotated[bool, typer.Option("--all", help="Include stopped and crashed workers")] = False,
) -> None:
    """List active workers (use --all to include stopped/crashed)."""
    with _get_client() as client:
        result = client.list_workers(include_stopped=show_all)
    workers = result.get("workers", [])
    if not workers:
        console.print("[dim]No workers.[/dim]")
        return
    table = Table(title="Workers")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Status")
    table.add_column("PID")
    table.add_column("Host")
    table.add_column("Started")
    for w in workers:
        color = "green" if w["status"] == "running" else ("red" if w["status"] == "crashed" else "yellow")
        from datetime import datetime, timezone

        started = datetime.fromtimestamp(w["started_at"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(w["id"], f"[{color}]{w['status']}[/{color}]", str(w.get("pid") or "-"), w.get("host") or "", started)
    console.print(table)


@worker_app.command("log")
def worker_log(
    worker_id: Annotated[str, typer.Argument(help="Worker ID")],
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Tail the log")] = False,
) -> None:
    """Print (or tail) a worker's log."""
    with _get_client() as client:
        if not follow:
            text = client.worker_log(worker_id)
            console.print(text, markup=False, end="")
            return
        try:
            for line in client.follow_worker_log(worker_id):
                console.print(line, markup=False)
        except KeyboardInterrupt:
            pass


@app.command("configure")
def configure_command(
    server_url: Annotated[Optional[str], typer.Option("--server-url", help="API server URL")] = None,
    api_key: Annotated[Optional[str], typer.Option("--api-key", help="API key")] = None,
) -> None:
    """Save server URL + API key to ~/.config/rag/config.toml."""
    from rag.cli_config import CliConfig, save_cli_config, load_cli_config

    current = load_cli_config()
    if server_url is None:
        prompt_default = current.server_url or ""
        server_url = typer.prompt("Server URL", default=prompt_default)
    if api_key is None:
        api_key = typer.prompt("API key", hide_input=True)
    save_cli_config(CliConfig(server_url=server_url, api_key=api_key))
    console.print("[green]Saved CLI config.[/green]")


@app.command("_worker-run", hidden=True)
def _worker_run(
    worker_id: Annotated[str, typer.Option("--worker-id", help="Supervisor-assigned worker id")] = "",
    poll_interval: Annotated[int, typer.Option(help="Seconds between polls when queue is empty")] = 5,
    stuck_minutes: Annotated[int, typer.Option(help="Minutes before a processing job is considered stuck")] = 30,
) -> None:
    """Internal: long-running worker process spawned by the API supervisor."""
    from rag.worker import run_worker
    if worker_id:
        os_environ_safe_set("RAG_WORKER_ID", worker_id)
    run_worker(poll_interval=poll_interval, stuck_minutes=stuck_minutes)


def os_environ_safe_set(key: str, value: str) -> None:
    import os
    os.environ[key] = value


@app.command("retrieve")
def retrieve_command(
    query: Annotated[str, typer.Argument(help="Natural language query")],
    source_id: Annotated[
        Optional[list[str]],
        typer.Option("--source-id", help="Restrict retrieval to one or more source IDs"),
    ] = None,
    filter: Annotated[
        Optional[list[str]],
        typer.Option("--filter", help="Metadata filter as key=value"),
    ] = None,
    seed_count: Annotated[Optional[int], typer.Option(help="Root seed count override")] = None,
    result_count: Annotated[Optional[int], typer.Option(help="Final result count override")] = None,
    rrf_k: Annotated[Optional[int], typer.Option(help="RRF k override")] = None,
    entity_confidence_threshold: Annotated[
        Optional[float],
        typer.Option(help="Relationship confidence threshold override"),
    ] = None,
    first_hop_similarity_threshold: Annotated[
        Optional[float],
        typer.Option(help="First-hop similarity threshold override"),
    ] = None,
    second_hop_similarity_threshold: Annotated[
        Optional[float],
        typer.Option(help="Second-hop similarity threshold override"),
    ] = None,
    trace: Annotated[bool, typer.Option("--trace", help="Print retrieval activity to stdout")] = False,
) -> None:
    """Run retrieval and print the final JSON response."""
    try:
        parsed_filters = _parse_key_value_pairs(filter, "filter")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if _use_api():
        with _get_client() as client:
            response = client.retrieve(
                query,
                source_ids=source_id or None,
                filters=parsed_filters or None,
                seed_count=seed_count,
                result_count=result_count,
                rrf_k=rrf_k,
                entity_confidence_threshold=entity_confidence_threshold,
                first_hop_similarity_threshold=first_hop_similarity_threshold,
                second_hop_similarity_threshold=second_hop_similarity_threshold,
                trace=trace,
            )
        console.print_json(json.dumps(response))
        return

    trace_printer = None
    if trace:
        def _printer(message: str) -> None:
            console.print(f"[trace] {message}", markup=False)
        trace_printer = _printer

    response = retrieve(
        query=query,
        source_ids=source_id or [],
        filters=parsed_filters,
        seed_count=seed_count,
        result_count=result_count,
        rrf_k=rrf_k,
        entity_confidence_threshold=entity_confidence_threshold,
        first_hop_similarity_threshold=first_hop_similarity_threshold,
        second_hop_similarity_threshold=second_hop_similarity_threshold,
        trace=trace,
        trace_printer=trace_printer,
    )
    console.print_json(json.dumps(response))


@app.command("search")
def search_command(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum number of results")] = settings.SEARCH_DEFAULT_LIMIT,
    min_score: Annotated[float, typer.Option("--min-score", help="Minimum score threshold")] = settings.SEARCH_MIN_SCORE,
) -> None:
    """Hybrid search over chunks and insights and return ranked results as JSON."""
    if _use_api():
        with _get_client() as client:
            payload = client.search(query, limit=limit, min_score=min_score)
        console.print_json(json.dumps(payload.get("results", payload)))
        return
    results = hybrid_search(query, limit=limit, min_score=min_score)
    console.print_json(
        json.dumps(
            {
                "chunks": [
                    {
                        "score": r.score,
                        "chunk": r.chunk,
                        "chunk_id": r.chunk_id,
                        "source_id": r.source_id,
                        "source_path": r.source_path,
                        "source_metadata": r.source_metadata,
                    }
                    for r in results.chunks
                ],
                "insights": [
                    {
                        "score": r.score,
                        "insight": r.insight,
                        "insight_id": r.insight_id,
                        "topics": r.topics,
                        "sources": [
                            {
                                "source_id": s.source_id,
                                "source_path": s.source_path,
                                "source_metadata": s.source_metadata,
                            }
                            for s in r.sources
                        ],
                    }
                    for r in results.insights
                ],
            }
        )
    )


@app.command("source")
def source_command(
    source_id: Annotated[str, typer.Argument(help="Source UUID")],
) -> None:
    """Print the stored markdown for a source."""
    if _use_api():
        from rag.api_client import ApiError
        with _get_client() as client:
            try:
                detail = client.get_source(source_id)
            except ApiError as e:
                if e.status == 404:
                    console.print(f"[red]Source not found: {source_id}[/red]")
                    raise typer.Exit(1)
                raise
        console.print(detail.get("markdown_content") or "", markup=False)
        return

    from rag.sources import get_source_detail

    detail = get_source_detail(source_id, connection_factory=_get_connection)
    if not detail:
        console.print(f"[red]Source not found: {source_id}[/red]")
        raise typer.Exit(1)

    console.print(detail["markdown_content"] or "", markup=False)


@sources_app.command("list")
def sources_list() -> None:
    """List all active sources."""
    if _use_api():
        with _get_client() as client:
            data = client.list_sources(limit=100, offset=0)
        rows_for_api = data.get("sources", [])
        if not rows_for_api:
            console.print("[dim]No sources found.[/dim]")
            return
        table = Table(title="Sources")
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Name")
        table.add_column("File")
        table.add_column("Type")
        table.add_column("Created")
        for r in rows_for_api:
            table.add_row(
                r.get("source_id", ""),
                r.get("name") or "",
                r.get("file_name") or "",
                r.get("file_type") or "",
                (r.get("created_at") or "")[:19],
            )
        console.print(table)
        return
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, file_name, file_type, version, created_at
            FROM sources
            WHERE deleted_at IS NULL
            ORDER BY created_at DESC
            """
        ).fetchall()

    if not rows:
        console.print("[dim]No sources found.[/dim]")
        return

    table = Table(title="Sources")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name")
    table.add_column("File")
    table.add_column("Type")
    table.add_column("Ver")
    table.add_column("Created")
    for r in rows:
        table.add_row(
            str(r[0]),
            r[1] or "",
            r[2] or "",
            r[3] or "",
            str(r[4]),
            str(r[5])[:19],
        )
    console.print(table)


@sources_app.command("get")
def sources_get(
    source_id: Annotated[str, typer.Argument(help="Source UUID")],
) -> None:
    """Show source details and markdown preview."""
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, file_name, file_type, storage_path, md5, version,
                   metadata, markdown_content, created_at
            FROM sources
            WHERE id = %s AND deleted_at IS NULL
            """,
            (source_id,),
        ).fetchone()
        insight_row = conn.execute(
            """
            SELECT COUNT(DISTINCT ci.insight_id)
            FROM chunks c
            JOIN chunk_insights ci ON ci.chunk_id = c.id
            WHERE c.source_id = %s AND c.deleted_at IS NULL
            """,
            (source_id,),
        ).fetchone()

    if not row:
        console.print(f"[red]Source not found: {source_id}[/red]")
        raise typer.Exit(1)

    insight_count = int(insight_row[0]) if insight_row else 0
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    fields = [
        ("ID", str(row[0])),
        ("Name", row[1] or ""),
        ("File", row[2] or ""),
        ("Type", row[3] or ""),
        ("Storage path", row[4] or ""),
        ("MD5", row[5] or ""),
        ("Version", str(row[6])),
        ("Metadata", str(row[7]) if row[7] else "{}"),
        ("Created", str(row[9])[:19]),
        ("Insights extracted", "Yes" if insight_count > 0 else "No"),
        ("Insight count", str(insight_count)),
    ]
    for k, v in fields:
        table.add_row(k, v)
    console.print(Panel(table, title=f"[bold]Source {source_id}[/bold]"))

    if row[8]:
        preview = row[8][:500] + ("…" if len(row[8]) > 500 else "")
        console.print(Panel(preview, title="Markdown preview"))
    else:
        console.print("[dim]No markdown content.[/dim]")


@sources_app.command("insights")
def sources_insights(
    source_id: Annotated[str, typer.Argument(help="Source UUID")],
) -> None:
    """List insights extracted from chunks of a source."""
    if _use_api():
        with _get_client() as client:
            data = client.source_insights(source_id)
        items = [
            {
                "id": i.get("insight_id", ""),
                "content": i.get("insight", ""),
                "topics": i.get("topics", []),
            }
            for i in data.get("insights", [])
        ]
        console.print_json(json.dumps(items))
        return
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT i.id, i.content, ci.topics, i.created_at
                FROM chunks c
                JOIN chunk_insights ci ON ci.chunk_id = c.id
                JOIN insights i ON i.id = ci.insight_id
                WHERE c.source_id = %s AND c.deleted_at IS NULL
                ORDER BY i.created_at
                """,
                (source_id,),
            )
            rows = cur.fetchall()

    console.print_json(
        json.dumps(
            [
                {
                    "id": str(insight_id),
                    "content": content or "",
                    "topics": list(topics or []),
                }
                for insight_id, content, topics, _created_at in rows
            ]
        )
    )


@sources_app.command("last")
def sources_last(
    k: Annotated[
        str,
        typer.Argument(help="Integer for last N sources, or date string for sources since date"),
    ],
) -> None:
    """Print source IDs for the last N sources or sources since a date."""
    with _get_connection() as conn:
        with conn.cursor() as cur:
            try:
                n = int(k)
            except ValueError:
                cur.execute(
                    """
                    SELECT id
                    FROM sources
                    WHERE deleted_at IS NULL AND created_at >= %s::timestamptz
                    ORDER BY created_at DESC
                    """,
                    (k,),
                )
            else:
                cur.execute(
                    """
                    SELECT id
                    FROM sources
                    WHERE deleted_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (n,),
                )
            rows = cur.fetchall()

    if not rows:
        console.print("[dim]No sources found.[/dim]")
        return

    for (source_id,) in rows:
        console.print(str(source_id))


@sources_app.command("search")
def sources_search(
    query: Annotated[
        str,
        typer.Argument(
            help="Search term. Use 'key:value' to match a specific metadata key, or a bare value to match any key, name, or file name."
        ),
    ],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum number of results")] = 20,
) -> None:
    """Search sources by metadata value (fuzzy substring match, no embeddings)."""
    if _use_api():
        with _get_client() as client:
            result = client.list_sources(limit=limit, q=query)
        rows = result["sources"]
    else:
        from rag.sources import list_recent_sources

        result = list_recent_sources(limit=limit, q=query, connection_factory=_get_connection)
        rows = result["sources"]

    if not rows:
        console.print("[dim]No sources matched.[/dim]")
        return

    table = Table(title=f"Sources matching {query!r}")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name")
    table.add_column("File")
    table.add_column("Type")
    table.add_column("Metadata")
    table.add_column("Created")
    table.add_column("Insights", justify="right")
    for r in rows:
        table.add_row(
            r["source_id"],
            r["name"] or "",
            r["file_name"] or "",
            r["file_type"] or "",
            str(r["metadata"]) if r["metadata"] else "{}",
            str(r["created_at"])[:19],
            str(r["insight_count"]),
        )
    console.print(table)
    if result["total"] > limit:
        console.print(f"[dim]Showing {limit} of {result['total']} matches.[/dim]")


@sources_app.command("delete")
def sources_delete(
    source_id: Annotated[str, typer.Argument(help="Source UUID")],
    hard: Annotated[bool, typer.Option("--hard", help="Hard-delete: remove file from disk")] = False,
) -> None:
    """Delete a source (soft by default, hard with --hard)."""
    if _use_api():
        from rag.api_client import ApiError
        with _get_client() as client:
            try:
                client.delete_source(source_id, hard=hard)
            except ApiError as e:
                if e.status == 404:
                    console.print(f"[red]Source not found: {source_id}[/red]")
                    raise typer.Exit(1)
                raise
        if hard:
            console.print(f"[green]Hard-deleted source {source_id} (DB records and file removed).[/green]")
        else:
            console.print(f"[green]Soft-deleted source {source_id}.[/green]")
        return

    from rag.ingestion import _write_audit_log, delete_source_artifacts
    from rag.storage import delete_stored_file

    with _get_connection() as conn:
        row = conn.execute(
            "SELECT storage_path FROM sources WHERE id = %s AND deleted_at IS NULL",
            (source_id,),
        ).fetchone()

        if not row:
            console.print(f"[red]Source not found: {source_id}[/red]")
            raise typer.Exit(1)

        if hard:
            with _get_graph_driver() as driver:
                delete_source_artifacts(conn, driver, source_id)
        else:
            conn.execute(
                "UPDATE sources SET deleted_at = now() WHERE id = %s",
                (source_id,),
            )
        _write_audit_log(
            conn,
            "source_hard_deleted" if hard else "source_soft_deleted",
            "source",
            source_id,
            {"hard": hard},
        )
        conn.commit()

    if hard:
        delete_stored_file(source_id)
        console.print(f"[green]Hard-deleted source {source_id} (DB records and file removed).[/green]")
    else:
        console.print(f"[green]Soft-deleted source {source_id}.[/green]")


@jobs_app.command("list")
def jobs_list(
    status: Annotated[Optional[str], typer.Option("--status", help="Filter by status")] = None,
    stats: Annotated[bool, typer.Option("--stats", help="Show job counts by status")] = False,
    retry: Annotated[bool, typer.Option("--retry", help="Retry all failed jobs")] = False,
) -> None:
    """List ingestion jobs."""
    if _use_api():
        with _get_client() as client:
            if stats:
                data = client.job_stats()
                stats_rows = [(s["status"], s["count"]) for s in data.get("stats", [])]
                if not stats_rows:
                    console.print("[dim]No jobs found.[/dim]")
                    return
                table = Table(title="Job Stats")
                table.add_column("Status")
                table.add_column("Count", justify="right")
                for status_group, cnt in stats_rows:
                    color = "green" if status_group == "completed" else ("red" if status_group == "failed" else "yellow")
                    table.add_row(f"[{color}]{status_group}[/{color}]", str(cnt))
                console.print(table)
                return
            if retry:
                failed = client.list_jobs(status="failed").get("jobs", [])
                if not failed:
                    console.print("[dim]No failed jobs found.[/dim]")
                    return
                retried = 0
                for j in failed:
                    try:
                        client.retry_job(j["id"])
                        retried += 1
                    except Exception as e:
                        console.print(f"[yellow]Could not retry {j['id']}: {e}[/yellow]")
                label = "job" if retried == 1 else "jobs"
                console.print(f"[green]{retried} {label} submitted for retry.[/green]")
                return
            data = client.list_jobs(status=status)
            jobs = data.get("jobs", [])
            if not jobs:
                console.print("[dim]No jobs found.[/dim]")
                return
            table = Table(title="Jobs")
            table.add_column("ID", style="dim", no_wrap=True)
            table.add_column("Source ID", style="dim")
            table.add_column("Status")
            table.add_column("Stage")
            table.add_column("Created")
            for j in jobs:
                status_color = "green" if j["status"] == "completed" else ("red" if j["status"].startswith("failed") else "yellow")
                table.add_row(j["id"], j.get("source_id") or "", f"[{status_color}]{j['status']}[/{status_color}]", j.get("current_stage") or "", (j.get("created_at") or "")[:19])
            console.print(table)
            return

    if stats:
        with _get_connection() as conn:
            rows = conn.execute(
                """SELECT
                     CASE
                       WHEN status LIKE 'failed:%' THEN 'failed'
                       WHEN status LIKE 'processing:%' THEN 'processing'
                       ELSE status
                     END AS status_group,
                     COUNT(*) AS cnt
                   FROM jobs
                   GROUP BY status_group
                   ORDER BY status_group"""
            ).fetchall()
        if not rows:
            console.print("[dim]No jobs found.[/dim]")
            return
        table = Table(title="Job Stats")
        table.add_column("Status")
        table.add_column("Count", justify="right")
        for status_group, cnt in rows:
            color = "green" if status_group == "completed" else ("red" if status_group == "failed" else "yellow")
            table.add_row(f"[{color}]{status_group}[/{color}]", str(cnt))
        console.print(table)
        return

    if retry:
        with _get_connection() as conn:
            failed_rows = conn.execute(
                "SELECT id FROM jobs WHERE status LIKE 'failed:%'"
            ).fetchall()
        if not failed_rows:
            console.print("[dim]No failed jobs found.[/dim]")
            return
        retried = 0
        for (job_id,) in failed_rows:
            try:
                retry_job(str(job_id))
                retried += 1
            except Exception as e:
                console.print(f"[yellow]Could not retry {job_id}: {e}[/yellow]")
        label = "job" if retried == 1 else "jobs"
        console.print(f"[green]{retried} {label} submitted for retry.[/green]")
        return

    with _get_connection() as conn:
        if status:
            if status in ("failed", "processing"):
                rows = conn.execute(
                    "SELECT id, source_id, status, current_stage, stage_log, created_at, updated_at FROM jobs WHERE status LIKE %s ORDER BY created_at DESC",
                    (f"{status}:%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, source_id, status, current_stage, stage_log, created_at, updated_at FROM jobs WHERE status = %s ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, source_id, status, current_stage, stage_log, created_at, updated_at FROM jobs ORDER BY updated_at DESC LIMIT 50"
            ).fetchall()

    if not rows:
        console.print("[dim]No jobs found.[/dim]")
        return

    table = Table(title="Jobs")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Source ID", style="dim")
    table.add_column("Status")
    table.add_column("Stage")
    table.add_column("Created")
    for r in rows:
        status_color = "green" if r[2] == "completed" else ("red" if r[2].startswith("failed") else "yellow")
        table.add_row(str(r[0]), str(r[1]), f"[{status_color}]{r[2]}[/{status_color}]", r[3] or "", str(r[5])[:19])
    console.print(table)


@jobs_app.command("status")
def jobs_status(
    job_id: Annotated[str, typer.Argument(help="Job UUID")],
) -> None:
    """Show job details and stage log."""
    if _use_api():
        from rag.api_client import ApiError
        with _get_client() as client:
            try:
                j = client.get_job(job_id)
            except ApiError as e:
                if e.status == 404:
                    console.print(f"[red]Job not found: {job_id}[/red]")
                    raise typer.Exit(1)
                raise
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        for k, v in [
            ("Job ID", j["id"]),
            ("Source ID", j.get("source_id") or ""),
            ("Status", j["status"]),
            ("Current Stage", j.get("current_stage") or ""),
            ("Created", (j.get("created_at") or "")[:19]),
            ("Updated", (j.get("updated_at") or "")[:19]),
        ]:
            table.add_row(k, v)
        console.print(Panel(table, title=f"[bold]Job {job_id}[/bold]"))
        if j.get("stage_log"):
            console.print(Panel(json.dumps(j["stage_log"], indent=2), title="Stage Log"))
        if j.get("error_detail"):
            console.print(Panel(json.dumps(j["error_detail"], indent=2), title="[red]Error Detail[/red]"))
        return

    with _get_connection() as conn:
        row = conn.execute(
            "SELECT id, source_id, status, current_stage, stage_log, created_at, updated_at, error_detail FROM jobs WHERE id = %s",
            (job_id,),
        ).fetchone()

    if not row:
        console.print(f"[red]Job not found: {job_id}[/red]")
        raise typer.Exit(1)

    table = Table(show_header=False, box=None)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for k, v in [
        ("Job ID", str(row[0])),
        ("Source ID", str(row[1])),
        ("Status", str(row[2])),
        ("Current Stage", str(row[3])),
        ("Created", str(row[5])[:19]),
        ("Updated", str(row[6])[:19]),
    ]:
        table.add_row(k, v)
    console.print(Panel(table, title=f"[bold]Job {job_id}[/bold]"))

    if row[4]:
        import json
        console.print(Panel(json.dumps(row[4], indent=2), title="Stage Log"))

    if row[7]:
        import json
        console.print(Panel(json.dumps(row[7], indent=2), title="[red]Error Detail[/red]"))


@jobs_app.command("retry")
def jobs_retry(
    job_id: Annotated[str, typer.Argument(help="Job UUID to retry")],
    from_stage: Annotated[
        Optional[str],
        typer.Option("--from-stage", help=f"Stage to retry from: {STAGE_ORDER}"),
    ] = None,
) -> None:
    """Retry a failed job."""
    if _use_api():
        from rag.api_client import ApiError
        with _get_client() as client:
            try:
                result = client.retry_job(job_id, from_stage=from_stage)
            except ApiError as e:
                console.print(f"[red]{e.detail}[/red]")
                raise typer.Exit(1)
        console.print(f"[green]Job {result['job_id']} queued for retry from stage '{result['retry_from_stage']}'.[/green]")
        return
    try:
        result = retry_job(job_id, from_stage=from_stage)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Retry failed: {e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Job {result['job_id']} queued for retry from stage '{result['retry_from_stage']}'.[/green]")


@jobs_app.command("cancel")
def jobs_cancel(
    job_id: Annotated[str, typer.Argument(help="Job UUID to cancel")],
) -> None:
    """Cancel a pending or processing job."""
    if _use_api():
        from rag.api_client import ApiError
        with _get_client() as client:
            try:
                result = client.cancel_job(job_id)
            except ApiError as e:
                console.print(f"[yellow]{e.detail}[/yellow]")
                raise typer.Exit(1)
        console.print(f"[green]Job {result['job_id']} has been cancelled.[/green]")
        return
    try:
        result = cancel_job(job_id)
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Cancellation failed: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Job {result['job_id']} has been cancelled.[/green]")


@community_app.command("ids")
def community_ids(
    source_id: Annotated[list[str], typer.Argument(help="Source IDs to scope")],
    semantic_threshold: Annotated[Optional[float], typer.Option("--semantic-threshold", help="Cosine similarity threshold for semantic edges")] = None,
    cutoff: Annotated[Optional[float], typer.Option("--cutoff", help="Minimum chunk score")] = None,
    min_community_size: Annotated[Optional[int], typer.Option("--min-community-size", help="Minimum entities per community")] = None,
    top_k: Annotated[Optional[int], typer.Option("--top-k", help="Max chunks per community")] = None,
    summarize: Annotated[Optional[str], typer.Option("--summarize", help="Model name to summarize communities")] = None,
    cross_source_top_k: Annotated[Optional[int], typer.Option("--cross-source-top-k", help="Max cross-source ANN neighbors per entity")] = None,
    max_cross_source_queries: Annotated[Optional[int], typer.Option("--max-cross-source-queries", help="Hard cap on per-entity ANN queries")] = None,
) -> None:
    """Detect communities from explicit source IDs."""
    if _use_api():
        with _get_client() as client:
            payload = {
                "scope_mode": "ids",
                "source_ids": list(source_id),
                "community_options": {
                    k: v for k, v in {
                        "semantic_threshold": semantic_threshold,
                        "cutoff": cutoff,
                        "min_community_size": min_community_size,
                        "top_k_chunks": top_k,
                        "cross_source_top_k": cross_source_top_k,
                        "max_cross_source_queries": max_cross_source_queries,
                    }.items() if v is not None
                },
            }
            if summarize:
                payload["summarize_model"] = summarize
            result = client.community(payload)
        console.print_json(json.dumps(result))
        return
    result = detect_communities(
        scope_mode="ids", source_ids=list(source_id), criteria=[], filters={},
        search_options={}, retrieve_options={},
        semantic_threshold=semantic_threshold, cutoff=cutoff,
        min_community_size=min_community_size, top_k_chunks=top_k,
        summarize_model=summarize,
        cross_source_top_k=cross_source_top_k,
        max_cross_source_queries=max_cross_source_queries,
    )
    console.print_json(json.dumps(result))


@community_app.command("search")
def community_search(
    criteria: Annotated[list[str], typer.Argument(help="Search criteria strings")],
    filter: Annotated[Optional[list[str]], typer.Option("--filter", help="Metadata filter key=value")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max results per criterion")] = settings.SEARCH_DEFAULT_LIMIT,
    min_score: Annotated[float, typer.Option("--min-score", help="Min search score")] = settings.SEARCH_MIN_SCORE,
    semantic_threshold: Annotated[Optional[float], typer.Option("--semantic-threshold")] = None,
    cutoff: Annotated[Optional[float], typer.Option("--cutoff")] = None,
    min_community_size: Annotated[Optional[int], typer.Option("--min-community-size")] = None,
    top_k: Annotated[Optional[int], typer.Option("--top-k")] = None,
    summarize: Annotated[Optional[str], typer.Option("--summarize")] = None,
    cross_source_top_k: Annotated[Optional[int], typer.Option("--cross-source-top-k", help="Max cross-source ANN neighbors per entity")] = None,
    max_cross_source_queries: Annotated[Optional[int], typer.Option("--max-cross-source-queries", help="Hard cap on per-entity ANN queries")] = None,
) -> None:
    """Detect communities from sources matched by search criteria."""
    try:
        parsed_filters = _parse_key_value_pairs(filter, "filter")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    if _use_api():
        with _get_client() as client:
            payload = {
                "scope_mode": "search",
                "criteria": list(criteria),
                "filters": parsed_filters,
                "search_options": {"limit": limit, "min_score": min_score},
                "community_options": {
                    k: v for k, v in {
                        "semantic_threshold": semantic_threshold,
                        "cutoff": cutoff,
                        "min_community_size": min_community_size,
                        "top_k_chunks": top_k,
                        "cross_source_top_k": cross_source_top_k,
                        "max_cross_source_queries": max_cross_source_queries,
                    }.items() if v is not None
                },
            }
            if summarize:
                payload["summarize_model"] = summarize
            result = client.community(payload)
        console.print_json(json.dumps(result))
        return
    result = detect_communities(
        scope_mode="search", source_ids=[], criteria=list(criteria),
        filters=parsed_filters, search_options={"limit": limit, "min_score": min_score},
        retrieve_options={}, semantic_threshold=semantic_threshold, cutoff=cutoff,
        min_community_size=min_community_size, top_k_chunks=top_k, summarize_model=summarize,
        cross_source_top_k=cross_source_top_k,
        max_cross_source_queries=max_cross_source_queries,
    )
    console.print_json(json.dumps(result))


@community_app.command("retrieve")
def community_retrieve(
    criteria: Annotated[list[str], typer.Argument(help="Retrieval criteria strings")],
    filter: Annotated[Optional[list[str]], typer.Option("--filter")] = None,
    seed_count: Annotated[Optional[int], typer.Option("--seed-count")] = None,
    result_count: Annotated[Optional[int], typer.Option("--result-count")] = None,
    rrf_k: Annotated[Optional[int], typer.Option("--rrf-k")] = None,
    entity_confidence_threshold: Annotated[Optional[float], typer.Option("--entity-confidence-threshold")] = None,
    first_hop_similarity_threshold: Annotated[Optional[float], typer.Option("--first-hop-similarity-threshold")] = None,
    second_hop_similarity_threshold: Annotated[Optional[float], typer.Option("--second-hop-similarity-threshold")] = None,
    trace: Annotated[bool, typer.Option("--trace")] = False,
    semantic_threshold: Annotated[Optional[float], typer.Option("--semantic-threshold")] = None,
    cutoff: Annotated[Optional[float], typer.Option("--cutoff")] = None,
    min_community_size: Annotated[Optional[int], typer.Option("--min-community-size")] = None,
    top_k: Annotated[Optional[int], typer.Option("--top-k")] = None,
    summarize: Annotated[Optional[str], typer.Option("--summarize")] = None,
    cross_source_top_k: Annotated[Optional[int], typer.Option("--cross-source-top-k", help="Max cross-source ANN neighbors per entity")] = None,
    max_cross_source_queries: Annotated[Optional[int], typer.Option("--max-cross-source-queries", help="Hard cap on per-entity ANN queries")] = None,
) -> None:
    """Detect communities from sources matched by retrieve criteria."""
    try:
        parsed_filters = _parse_key_value_pairs(filter, "filter")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    if _use_api():
        with _get_client() as client:
            payload = {
                "scope_mode": "retrieve",
                "criteria": list(criteria),
                "filters": parsed_filters,
                "retrieve_options": {
                    k: v for k, v in {
                        "seed_count": seed_count,
                        "result_count": result_count,
                        "rrf_k": rrf_k,
                        "entity_confidence_threshold": entity_confidence_threshold,
                        "first_hop_similarity_threshold": first_hop_similarity_threshold,
                        "second_hop_similarity_threshold": second_hop_similarity_threshold,
                        "trace": trace,
                    }.items() if v is not None
                },
                "community_options": {
                    k: v for k, v in {
                        "semantic_threshold": semantic_threshold,
                        "cutoff": cutoff,
                        "min_community_size": min_community_size,
                        "top_k_chunks": top_k,
                        "cross_source_top_k": cross_source_top_k,
                        "max_cross_source_queries": max_cross_source_queries,
                    }.items() if v is not None
                },
            }
            if summarize:
                payload["summarize_model"] = summarize
            result = client.community(payload)
        console.print_json(json.dumps(result))
        return
    result = detect_communities(
        scope_mode="retrieve", source_ids=[], criteria=list(criteria),
        filters=parsed_filters, search_options={},
        retrieve_options={
            "seed_count": seed_count, "result_count": result_count, "rrf_k": rrf_k,
            "entity_confidence_threshold": entity_confidence_threshold,
            "first_hop_similarity_threshold": first_hop_similarity_threshold,
            "second_hop_similarity_threshold": second_hop_similarity_threshold,
            "trace": trace,
        },
        semantic_threshold=semantic_threshold, cutoff=cutoff,
        min_community_size=min_community_size, top_k_chunks=top_k, summarize_model=summarize,
        cross_source_top_k=cross_source_top_k,
        max_cross_source_queries=max_cross_source_queries,
    )
    console.print_json(json.dumps(result))


if __name__ == "__main__":
    app()
