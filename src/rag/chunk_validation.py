import json
import logging
import math
import random
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

import requests

from rag import jev, prompts
from rag.chunking import ChunkData
from rag.config import settings

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Jev backend (CHUNK_VALIDATION_BACKEND=jev): score EVERY chunk, skip the bad
# ones (soft delete), never fail the job for quality reasons.
# ---------------------------------------------------------------------------

# Characters of a chunk sent as Jev state. Chunks are normally a few hundred
# tokens; this only guards the 32k-token limit for oversized parent chunks.
JEV_MAX_CHUNK_CHARS = 12000

QUALITY_CRITERIA = ("complete", "coherent", "substantive")


def build_quality_questions() -> dict[str, dict]:
    """Three Nouls scoring one chunk (the Jev state). Jev reads questions
    literally, so each carries explicit true/false criteria."""
    return {
        "complete": jev.noul(
            "Does the text contain at least one complete, self-contained thought?",
            true=(
                "At least one full statement, explanation or item that makes sense on its "
                "own. Complete tables, lists, code and Q&A entries count."
            ),
            false=(
                "Only a cut-off fragment, a dangling clause or a few stray words that do "
                "not express a whole thought."
            ),
        ),
        "coherent": jev.noul(
            "Is the text internally coherent and readable?",
            true=(
                "The text reads as meaningful, well-formed content: prose, a table, a "
                "list, code or a transcript passage with a clear thread."
            ),
            false=(
                "The text is garbled, scrambled, corrupted by extraction errors, or a "
                "random jumble of unrelated fragments."
            ),
        ),
        "substantive": jev.noul(
            "Does the text carry information that could help answer someone's questions?",
            true=(
                "It states facts, explanations, arguments, data, instructions or "
                "examples about some subject."
            ),
            false=(
                "It is boilerplate: navigation menus, legal footers or copyright "
                "notices, a table of contents, sponsor or advertisement reads, page "
                "headers, or headings with no content under them."
            ),
        ),
    }


def quality_probabilities(answers: dict) -> dict[str, float]:
    """`{criterion: yes-probability}` from a Jev answers dict. Raises
    `KeyError`/`TypeError`/`ValueError` when an answer is missing or malformed."""
    return {name: jev.noul_p(answers[name]) for name in QUALITY_CRITERIA}


def min_quality_probability(answers: dict) -> float:
    return min(quality_probabilities(answers).values())


def select_skips(
    scores: Sequence[tuple[str, float | None]],
    protected_ids: Collection[str],
    *,
    threshold: float,
    max_skip_fraction: float,
) -> tuple[list[str], int, bool]:
    """Decide which chunks to skip.

    `scores` holds `(chunk_id, min_probability)`; `None` means the Jev call
    failed and the chunk passes (fail-open). A chunk is low quality when its
    score is below `threshold`. Returns `(skipped_ids, low_quality_count,
    cap_triggered)`.

    - Parents (`protected_ids`) are never skipped, but a low-quality parent
      still counts toward the low-quality fraction.
    - The fraction's denominator is the number of successfully scored chunks.
      If it exceeds `max_skip_fraction` the scoring is more likely wrong than
      the document, so NOTHING is skipped (`cap_triggered=True`). Skipping
      every scored chunk is also refused for the same reason.
    """
    scored = [(chunk_id, value) for chunk_id, value in scores if value is not None]
    low = [chunk_id for chunk_id, value in scored if value < threshold]
    if not low:
        return [], 0, False
    if len(low) / len(scored) > max_skip_fraction or len(low) == len(scored):
        return [], len(low), True
    protected = set(protected_ids)
    return [chunk_id for chunk_id in low if chunk_id not in protected], len(low), False


@dataclass
class JevValidationResult:
    validated_chunks: int
    skipped_chunk_ids: list[str] = field(default_factory=list)
    low_quality_chunks: int = 0
    jev_errors: int = 0
    cap_triggered: bool = False

    def stage_output(self) -> dict:
        return {
            "backend": "jev",
            "validated_chunks": self.validated_chunks,
            "skipped_chunks": list(self.skipped_chunk_ids),
            "low_quality_chunks": self.low_quality_chunks,
            "jev_errors": self.jev_errors,
            "cap_triggered": self.cap_triggered,
        }


def jev_validate_chunks(
    chunk_rows: Sequence[tuple[str, str]],
    protected_ids: Collection[str] = (),
) -> JevValidationResult:
    """Score every `(chunk_id, content)` row with one Jev call per chunk.

    Never raises for Jev reasons: a chunk whose call errored or returned a
    malformed answer passes; if every call errors nothing is skipped.
    """
    if not chunk_rows:
        return JevValidationResult(validated_chunks=0)

    questions = build_quality_questions()
    items = [(content[:JEV_MAX_CHUNK_CHARS], questions) for _, content in chunk_rows]
    try:
        results = jev.system_one_many(items, max_workers=settings.JEV_CONCURRENCY)
    except Exception as exc:  # defensive: system_one_many captures per-item errors
        log.warning("Jev chunk validation failed; skipping nothing: %s", exc)
        return JevValidationResult(validated_chunks=len(chunk_rows), jev_errors=len(chunk_rows))

    scores: list[tuple[str, float | None]] = []
    errors = 0
    first_error: Exception | None = None
    for (chunk_id, _), result in zip(chunk_rows, results):
        if isinstance(result, Exception):
            errors += 1
            first_error = first_error or result
            scores.append((chunk_id, None))
            continue
        try:
            scores.append((chunk_id, min_quality_probability(result)))
        except (KeyError, TypeError, ValueError) as exc:
            errors += 1
            first_error = first_error or exc
            scores.append((chunk_id, None))

    if errors:
        log.warning(
            "Jev chunk validation: %d/%d calls failed (first error: %s); failed chunks pass",
            errors, len(chunk_rows), first_error,
        )
    if errors == len(chunk_rows):
        return JevValidationResult(validated_chunks=len(chunk_rows), jev_errors=errors)

    skipped, low_count, cap_triggered = select_skips(
        scores,
        protected_ids,
        threshold=settings.CHUNK_VALIDATION_JEV_THRESHOLD,
        max_skip_fraction=settings.CHUNK_VALIDATION_MAX_SKIP_FRACTION,
    )
    if cap_triggered:
        log.warning(
            "Jev chunk validation: %d/%d chunks scored low quality (cap %.2f); "
            "skipping none, likely a scoring problem",
            low_count, len(chunk_rows) - errors, settings.CHUNK_VALIDATION_MAX_SKIP_FRACTION,
        )
    return JevValidationResult(
        validated_chunks=len(chunk_rows),
        skipped_chunk_ids=skipped,
        low_quality_chunks=low_count,
        jev_errors=errors,
        cap_triggered=cap_triggered,
    )
