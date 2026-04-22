def link_graph(conn, driver, source_id: str, job_id: str) -> None:
    """Compatibility stage kept for queued jobs; no graph mutations."""
    _ = (conn, driver, source_id, job_id)
