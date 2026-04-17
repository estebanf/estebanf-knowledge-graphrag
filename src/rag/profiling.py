import json
import re
from dataclasses import dataclass

import requests

from rag.config import settings

_SAMPLE_CHARS = 3000

_PROMPT = """\
You are a document analyst. Analyze the document excerpt below and classify it.

Return ONLY a valid JSON object with exactly these fields:
{
  "structure_type": "well-structured|loosely-structured|unstructured",
  "heading_consistency": "consistent|inconsistent|none",
  "content_density": "uniform|variable",
  "primary_content_type": "prose|tabular|mixed|qa_pairs|code|transcript",
  "avg_section_length": "short|medium|long",
  "has_tables": true|false,
  "has_code_blocks": true|false,
  "domain": "legal|financial|technical|general|medical|policy"
}

Document excerpt:
"""


@dataclass(frozen=True)
class DocumentProfile:
    structure_type: str
    heading_consistency: str
    content_density: str
    primary_content_type: str
    avg_section_length: str
    has_tables: bool
    has_code_blocks: bool
    domain: str


_DEFAULT_PROFILE = DocumentProfile(
    structure_type="unstructured",
    heading_consistency="none",
    content_density="uniform",
    primary_content_type="prose",
    avg_section_length="medium",
    has_tables=False,
    has_code_blocks=False,
    domain="general",
)


def profile_document(markdown: str) -> DocumentProfile:
    """Call OpenRouter to classify document structure. Returns default profile on any error."""
    if not settings.OPENROUTER_API_KEY:
        return _DEFAULT_PROFILE

    sample = markdown[:_SAMPLE_CHARS]
    payload = {
        "model": settings.MODEL_DOC_PROFILING,
        "messages": [{"role": "user", "content": _PROMPT + sample}],
        "temperature": 0,
        "max_tokens": 256,
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
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        return DocumentProfile(
            structure_type=data.get("structure_type", "unstructured"),
            heading_consistency=data.get("heading_consistency", "none"),
            content_density=data.get("content_density", "uniform"),
            primary_content_type=data.get("primary_content_type", "prose"),
            avg_section_length=data.get("avg_section_length", "medium"),
            has_tables=bool(data.get("has_tables", False)),
            has_code_blocks=bool(data.get("has_code_blocks", False)),
            domain=data.get("domain", "general"),
        )
    except Exception:
        return _DEFAULT_PROFILE
