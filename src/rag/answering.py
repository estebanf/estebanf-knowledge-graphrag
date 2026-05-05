import json
from collections.abc import Iterator

import requests

from rag import prompts
from rag.answer_models import get_supported_answer_models
from rag.config import settings
from rag.retrieval import retrieve


_OPENCODE_GO_CHAT_URL = "https://opencode.ai/zen/go/v1/chat/completions"
_OPENCODE_GO_MESSAGES_URL = "https://opencode.ai/zen/go/v1/messages"

_MINIMAX_MODELS = {"minimax-m2.5", "minimax-m2.7"}


def _require_api_key() -> None:
    if not (settings.OPENCODE_GO_API_KEY or settings.OPENCODE_API_KEY):
        raise ValueError("OPENCODE_GO_API_KEY or OPENCODE_API_KEY is required for answer generation")


def _opencode_go_key() -> str:
    return settings.OPENCODE_GO_API_KEY or settings.OPENCODE_API_KEY


def _opencode_go_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_opencode_go_key()}",
        "Content-Type": "application/json",
    }


def _is_anthropic_model(model: str) -> bool:
    return model in _MINIMAX_MODELS


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


def _stream_anthropic(response: requests.Response, retrieval_results: dict) -> Iterator[str]:
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        data = raw_line[6:]
        payload = json.loads(data)
        if payload.get("type") == "content_block_delta":
            text = payload.get("delta", {}).get("text")
            if text:
                yield _sse("answer_delta", {"delta": text})
        elif payload.get("type") == "message_stop":
            break

    yield _sse("results", retrieval_results)


def _stream_openai_compatible(response: requests.Response, retrieval_results: dict) -> Iterator[str]:
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        data = raw_line[6:]
        if data == "[DONE]":
            break
        payload = json.loads(data)
        delta = payload.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            yield _sse("answer_delta", {"delta": delta})

    yield _sse("results", retrieval_results)


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

    if _is_anthropic_model(model):
        url = _OPENCODE_GO_MESSAGES_URL
        body = {
            "model": model,
            "messages": [{"role": "user", "content": _build_answer_prompt(query, retrieval_results)}],
            "max_tokens": 4096,
            "stream": True,
        }
    else:
        url = _OPENCODE_GO_CHAT_URL
        body = {
            "model": model,
            "messages": [{"role": "user", "content": _build_answer_prompt(query, retrieval_results)}],
            "temperature": 0.2,
            "stream": True,
        }

    response = requests.post(
        url,
        headers=_opencode_go_headers(),
        json=body,
        timeout=300,
        stream=True,
    )
    response.raise_for_status()

    if _is_anthropic_model(model):
        yield from _stream_anthropic(response, retrieval_results)
    else:
        yield from _stream_openai_compatible(response, retrieval_results)
