import json
from collections.abc import Iterator

from rag import prompts
from rag.answer_models import get_supported_answer_models
from rag.config import settings
from rag.opencode import endpoint_for_model, new_session_id, stream_text
from rag.retrieval import retrieve


def _require_api_key() -> None:
    if not settings.OPENCODE_API_KEY:
        raise ValueError("OPENCODE_API_KEY is required for answer generation")


def _build_answer_prompt(query: str, results: dict) -> str:
    return (
        f'User question: "{query}"\n\n'
        + prompts.ANSWER_GENERATION
        + f"```\n{json.dumps(results, indent=2)}\n```"
    )


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


def _is_supported_model(model: str) -> bool:
    return any(item["id"] == model for item in get_supported_answer_models())


def stream_answer(
    *,
    query: str,
    model: str,
    source_ids: list[str],
    filters: dict[str, str],
    seed_count: int | None,
    result_count: int | None,
    rrf_k: int | None,
    entity_confidence_threshold: float | None,
    first_hop_similarity_threshold: float | None,
    second_hop_similarity_threshold: float | None,
) -> Iterator[str]:
    if not _is_supported_model(model):
        raise ValueError(f"Unsupported answer model: {model}")

    retrieval_results = retrieve(
        query=query,
        source_ids=source_ids,
        filters=filters,
        seed_count=seed_count,
        result_count=result_count,
        rrf_k=rrf_k,
        entity_confidence_threshold=entity_confidence_threshold,
        first_hop_similarity_threshold=first_hop_similarity_threshold,
        second_hop_similarity_threshold=second_hop_similarity_threshold,
        trace=False,
        trace_printer=None,
    )

    _require_api_key()

    # One session for this answer request.
    session_id = new_session_id("answer")
    prompt = _build_answer_prompt(query, retrieval_results)

    # Per-family request shape (reasoning effort for gpt-*, temperature for
    # chat-completions models, max_tokens for /messages) lives in rag.opencode.
    endpoint = endpoint_for_model(model)
    if endpoint == "/responses":
        # Reasoning tokens count against the output cap, so leave headroom.
        options = {"reasoning_effort": settings.ANSWER_REASONING_EFFORT, "max_tokens": 8192}
    elif endpoint == "/messages":
        options = {"max_tokens": 4096}
    else:
        options = {"temperature": 0.2}

    for delta in stream_text(model, prompt, session_id=session_id, **options):
        yield _sse("answer_delta", {"delta": delta})

    yield _sse("results", retrieval_results)
