import json
import math
import random
import re

import requests

from rag import prompts
from rag.chunking import ChunkData
from rag.config import settings

_FAILURE_THRESHOLD = 0.20


def _score_chunk(content: str) -> bool:
    """Returns True if chunk passes quality check. Returns True on any error."""
    payload = {
        "model": settings.MODEL_CHUNK_VALIDATION,
        "messages": [{"role": "user", "content": prompts.CHUNK_VALIDATION + content}],
        "temperature": 0,
        "max_tokens": 64,
    }
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        content_str = resp.json()["choices"][0]["message"]["content"].strip()
        content_str = re.sub(r"^```(?:json)?\s*", "", content_str)
        content_str = re.sub(r"\s*```$", "", content_str)
        return bool(json.loads(content_str).get("pass", True))
    except Exception:
        return True


def validate_chunks(
    chunks: list[ChunkData],
    domain: str,
    is_first_of_new_type: bool = False,
) -> bool:
    """Sample-based quality validation. Returns True if chunks pass, False to fail the job."""
    if not settings.OPENROUTER_API_KEY or not chunks:
        return True

    if is_first_of_new_type:
        sample_rate = 1.0
    elif domain in ("legal", "financial", "medical"):
        sample_rate = settings.CHUNK_VALIDATION_SAMPLE_RATE_HIGH_STAKES
    else:
        sample_rate = settings.CHUNK_VALIDATION_SAMPLE_RATE

    n = max(1, math.ceil(len(chunks) * sample_rate))
    sample = random.sample(chunks, min(n, len(chunks)))

    failures = sum(1 for c in sample if not _score_chunk(c.content))
    return (failures / len(sample)) <= _FAILURE_THRESHOLD
