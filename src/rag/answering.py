import json
from collections.abc import Iterator

import requests

from rag.answer_models import get_supported_answer_models
from rag.config import settings
from rag.retrieval import retrieve


_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def _require_api_key() -> None:
    if not settings.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is required for answer generation")


def _openrouter_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def _build_answer_prompt(query: str, results: dict) -> str:
    return (
        f'User question: "{query}"\n\n'
        "Use only the retrieved results below.\n\n"
        "Write a comprehensive narrative answer that clearly answers the question first, "
        "then elaborates on the supporting evidence.\n\n"
        "Requirements:\n"
        "1. Begin with a direct thesis that answers the exact question.\n"
        "2. Write 3 to 5 cohesive paragraphs, not bullets.\n"
        "3. Build the answer around the most relevant evidence, not around all major themes in the retrieval.\n"
        "4. Prioritize evidence that matches the narrow angle of the question. If the question mentions insurance, "
        "foreground insurance and cyber-risk evidence over general AI strategy.\n"
        "5. Use broader AI-native themes only when they strengthen the insurance positioning.\n"
        "6. Synthesize the evidence into a clear recommendation rather than listing retrieved points.\n"
        "7. Be explicit about what is directly supported by the retrieval, and avoid adding claims that are not grounded in it.\n"
        "8. If the retrieval contains conflicting signals, state the conflict instead of forcing a clean answer.\n"
        "9. End with a brief statement of what the retrieval does not establish, if there is an important gap.\n"
        "10. Do not mention chunk IDs, scores, or retrieval mechanics.\n"
        "11. Do not elevate a concept into the main answer just because it appears in a high-scoring chunk. "
        "Relevance to the exact question matters more than retrieval score.\n"
        "12. Avoid generic AI phrasing and avoid filler.\n\n"
        "Style guidance:\n"
        "- Sound analytical and well structured.\n"
        "- Be specific and concrete.\n"
        "- Prefer a strong narrative arc: answer, reasoning, implications, limitations.\n\n"
        "Retrieved results:\n"
        "```\n"
        f"{json.dumps(results, indent=2)}\n"
        "```"
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
    response = requests.post(
        _CHAT_COMPLETIONS_URL,
        headers=_openrouter_headers(),
        json={
            "model": model,
            "messages": [{"role": "user", "content": _build_answer_prompt(query, retrieval_results)}],
            "temperature": 0.2,
            "stream": True,
        },
        timeout=300,
        stream=True,
    )
    response.raise_for_status()

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
