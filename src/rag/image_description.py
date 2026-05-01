import base64

import requests

from rag.config import settings

_PROMPT = (
    "Describe this image concisely for use in a knowledge base. "
    "Focus on the key information, data, concepts, or visual content shown. "
    "Be factual and specific. Do not add interpretation beyond what is visible."
)


def describe_image(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.MODEL_IMAGE_DESCRIPTION,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        },
        timeout=(10, 60),
    )
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        err = body["error"]
        raise RuntimeError(f"OpenRouter error: {err.get('message', err)}")
    if not body.get("choices"):
        raise RuntimeError(f"Unexpected OpenRouter response: {body}")
    return body["choices"][0]["message"]["content"]
