import json
import re

import requests

from rag.config import settings

_SAMPLE_CHARS = 2000

_PROMPT = """\
You are a document classifier. Given the start of a document, extract the following metadata fields as JSON.

Fields:
- kind: one of transcript, paper, article, report, infographic, playbook, guide, policy, specification, other
- author: author name if detectable, otherwise null
- source: publication, company, or origin if detectable, otherwise null
- domain: one of legal, financial, technical, medical, general

Respond with ONLY a valid JSON object. No explanation. Example:
{"kind": "report", "author": "Jane Smith", "source": "Gartner", "domain": "technical"}

Document:
"""


def extract_metadata(markdown: str) -> dict:
    """Call OpenRouter to extract kind/author/source/domain from document markdown.

    Returns a dict with those four keys. On any error returns an empty dict
    so that ingestion can continue without failing.
    """
    if not settings.OPENROUTER_API_KEY:
        return {}

    sample = markdown[:_SAMPLE_CHARS]
    payload = {
        "model": settings.MODEL_METADATA_EXTRACTION,
        "messages": [{"role": "user", "content": _PROMPT + sample}],
        "temperature": 0,
        "max_tokens": 128,
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
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if model wraps the JSON
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        extracted = json.loads(content)
        return {
            k: extracted.get(k)
            for k in ("kind", "author", "source", "domain")
        }
    except Exception:
        return {}
