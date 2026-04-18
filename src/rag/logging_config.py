import logging

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON output with contextvars support.

    Call once at process startup (CLI entrypoint, worker entrypoint, etc.).
    After calling this, bind per-request context with::

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            job_id=job_id,
            source_id=source_id,
            api_key_name=api_key_name,
        )
        log = structlog.get_logger()
        log.info("stage_started", action="stage_start", stage="parsing")
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
