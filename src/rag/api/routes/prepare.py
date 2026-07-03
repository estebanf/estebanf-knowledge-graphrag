"""Prepare routes: backend-owned image description for CLI document preparation.

The CLI extracts image bytes from a binary document locally (via Docling) but
does not hold the OpenRouter key or image-model configuration (KTD2/R3). It asks
the backend to describe each image through this endpoint, which reuses the
server-side ``describe_image`` helper. The route is gated by the ``ingest`` scope
and validates the MIME type and payload size before making any LLM call.
"""

from __future__ import annotations

import base64
import binascii
import logging

from fastapi import APIRouter, Depends, HTTPException

from rag.api.auth import Principal, requires_scope
from rag.api.schemas import DescribeImageRequest, DescribeImageResponse
from rag.image_description import SUPPORTED_IMAGE_MIME_TYPES, describe_image

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prepare", tags=["prepare"])

# Cap the decoded image so the endpoint can't be used to funnel arbitrarily large
# payloads to the LLM. 10 MB comfortably covers Docling-extracted figures.
MAX_IMAGE_BYTES = 10 * 1024 * 1024
# base64 inflates by ~4/3; bound the encoded string before decoding it.
_MAX_BASE64_CHARS = (MAX_IMAGE_BYTES // 3 + 1) * 4


@router.post("/describe-image", response_model=DescribeImageResponse)
def describe_image_route(
    payload: DescribeImageRequest,
    principal: Principal = Depends(requires_scope("ingest")),
) -> DescribeImageResponse:
    """Describe a transient base64 image. The image is not stored."""
    if payload.mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=415, detail=f"unsupported image mime type: {payload.mime_type}"
        )
    if len(payload.image_base64) > _MAX_BASE64_CHARS:
        raise HTTPException(status_code=413, detail="image exceeds maximum size")

    try:
        image_bytes = base64.b64decode(payload.image_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="invalid base64 image content")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image content")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds maximum size")

    try:
        description = describe_image(image_bytes, payload.mime_type)
    except Exception:
        # Don't surface the upstream exception text (could echo request/config
        # detail); log it server-side and return a generic error.
        log.exception("image description failed")
        raise HTTPException(status_code=502, detail="image description failed")

    return DescribeImageResponse(description=description)
