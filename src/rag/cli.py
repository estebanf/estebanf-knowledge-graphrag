from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.db import get_connection
from rag.ingestion import ingest_file, retry_job, STAGE_ORDER, submit_ingestion_job, _write_audit_log
from rag.storage import delete_stored_file

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

app = typer.Typer(help="RAG CLI — document ingestion and management")
sources_app = typer.Typer(help="Manage ingested sources")
jobs_app = typer.Typer(help="Manage ingestion jobs")
app.add_typer(sources_app, name="sources")
app.add_typer(jobs_app, name="jobs")

console = Console()


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
    parsed_metadata: dict = {}
    for item in metadata or []:
        if "=" not in item:
            console.print(f"[red]Invalid metadata format: {item!r} (expected key=value)[/red]")
            raise typer.Exit(1)
        k, _, v = item.partition("=")
        parsed_metadata[k.strip()] = v.strip()

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
    for file in resolved_files:
        try:
            result = submit_ingestion_job(
                file,
                name=name if len(resolved_files) == 1 else None,
                metadata=parsed_metadata,
            )
            results.append((file.name, result["job_id"], result["status"]))
        except FileNotFoundError as e:
            errors.append((file.name, str(e)))
        except ValueError as e:
            errors.append((file.name, str(e)))
        except Exception as e:
            errors.append((file.name, f"Error: {e}"))

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


@app.command()
def worker(
    poll_interval: Annotated[int, typer.Option(help="Seconds between polls when queue is empty")] = 5,
    stuck_minutes: Annotated[int, typer.Option(help="Minutes before a processing job is considered stuck")] = 30,
) -> None:
    """Start the ingestion worker. Processes pending jobs until Ctrl+C."""
    from rag.worker import run_worker
    run_worker(poll_interval=poll_interval, stuck_minutes=stuck_minutes)


@sources_app.command("list")
def sources_list() -> None:
    """List all active sources."""
    with get_connection() as conn:
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
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, file_name, file_type, storage_path, md5, version,
                   metadata, markdown_content, created_at
            FROM sources
            WHERE id = %s AND deleted_at IS NULL
            """,
            (source_id,),
        ).fetchone()

    if not row:
        console.print(f"[red]Source not found: {source_id}[/red]")
        raise typer.Exit(1)

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
    ]
    for k, v in fields:
        table.add_row(k, v)
    console.print(Panel(table, title=f"[bold]Source {source_id}[/bold]"))

    if row[8]:
        preview = row[8][:500] + ("…" if len(row[8]) > 500 else "")
        console.print(Panel(preview, title="Markdown preview"))
    else:
        console.print("[dim]No markdown content.[/dim]")


@sources_app.command("delete")
def sources_delete(
    source_id: Annotated[str, typer.Argument(help="Source UUID")],
    hard: Annotated[bool, typer.Option("--hard", help="Hard-delete: remove file from disk")] = False,
) -> None:
    """Delete a source (soft by default, hard with --hard)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT storage_path FROM sources WHERE id = %s AND deleted_at IS NULL",
            (source_id,),
        ).fetchone()

        if not row:
            console.print(f"[red]Source not found: {source_id}[/red]")
            raise typer.Exit(1)

        if hard:
            conn.execute("DELETE FROM jobs WHERE source_id = %s", (source_id,))
            conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))
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
) -> None:
    """List ingestion jobs."""
    with get_connection() as conn:
        if status:
            rows = conn.execute(
                "SELECT id, source_id, status, current_stage, stage_log, created_at, updated_at FROM jobs WHERE status = %s ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, source_id, status, current_stage, stage_log, created_at, updated_at FROM jobs ORDER BY created_at DESC LIMIT 50"
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
    with get_connection() as conn:
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
    try:
        result = retry_job(job_id, from_stage=from_stage)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Retry failed: {e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Job {result['job_id']} completed successfully.[/green]")


@jobs_app.command("cancel")
def jobs_cancel(
    job_id: Annotated[str, typer.Argument(help="Job UUID to cancel")],
) -> None:
    """Cancel a pending or processing job."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, source_id, status FROM jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        if not row:
            console.print(f"[red]Job not found: {job_id}[/red]")
            raise typer.Exit(1)

        status = row[2]
        if status == "completed" or status == "cancelled":
            console.print(f"[yellow]Job {job_id} cannot be cancelled (status: {status}).[/yellow]")
            raise typer.Exit(1)

        conn.execute(
            "UPDATE jobs SET status = 'cancelled', updated_at = now() WHERE id = %s",
            (job_id,),
        )
        conn.commit()

    console.print(f"[green]Job {job_id} has been cancelled.[/green]")
