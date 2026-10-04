"""Jev-backed document profiling + metadata classification (`PROFILING_BACKEND=jev`).

ONE System One call over a document excerpt answers every Choice needed by the
chunking pipeline (`DocumentProfile` fields) plus the metadata `kind` and
`domain`. Free-text metadata (author/source) is something Jev cannot produce,
so `extract_profile_and_metadata` runs the small author/source LLM call
concurrently with the Jev call.

Design notes:
- Jev is biased toward the FIRST option of a Choice, so every Choice is asked
  twice in the same call with the option order reversed (`<field>__a` /
  `<field>__b`); the two probability distributions are averaged and the argmax
  wins. Extra questions are nearly free: the state is billed once per call.
- `has_tables` / `has_code_blocks` are computed deterministically from the full
  markdown, never via Jev.
- Profile `domain` (legal|financial|technical|general|medical|policy) and
  metadata `domain` (legal|financial|technical|medical|general) are separate
  questions: the metadata domain feeds `/api/sources/facets` and frontend
  filters and must stay within its 5 values.
"""

from __future__ import annotations

import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from rag import jev
from rag.metadata_extraction import extract_author_source
from rag.profiling import DocumentProfile

log = logging.getLogger(__name__)

# Characters of the document sent to Jev (~1.5k tokens, far below the 32k limit).
EXCERPT_CHARS = 6000

# Profile choice fields: (instructions, ordered {option: description}).
# Option order here is the canonical "a" order; the "b" variant is reversed.
PROFILE_CHOICES: dict[str, tuple[str, dict[str, str]]] = {
    "structure_type": (
        "How well is this document organised into sections? Judge only the visible excerpt.",
        {
            "well-structured": "Clear sections with headings and a logical hierarchy throughout.",
            "loosely-structured": "Some headings, lists or sections, but the organisation is uneven or partial.",
            "unstructured": "Continuous text with no meaningful sections or headings.",
        },
    ),
    "heading_consistency": (
        "How consistently are markdown headings (lines starting with #) used in this excerpt?",
        {
            "consistent": "Headings are used throughout with a regular, orderly hierarchy.",
            "inconsistent": "Headings exist but are irregular: skipped levels, mixed styles or sporadic use.",
            "none": "There are no markdown headings in the excerpt.",
        },
    ),
    "content_density": (
        "Is the amount of content per section or paragraph even across the excerpt?",
        {
            "uniform": "Sections and paragraphs are of similar size and information density.",
            "variable": "Dense passages sit next to sparse ones, e.g. long sections next to one-liners.",
        },
    ),
    "primary_content_type": (
        "What is the dominant form of the content in this excerpt?",
        {
            "prose": "Flowing paragraphs of written text such as articles, reports or essays.",
            "tabular": "Mostly tables or data laid out in rows and columns.",
            "mixed": "A substantial mix of prose, tables, lists or code with no clear majority.",
            "qa_pairs": "Question and answer pairs, FAQ-style entries.",
            "code": "Mostly source code or command-line listings.",
            "transcript": "Spoken-word text such as an interview, meeting, podcast or video transcript.",
        },
    ),
    "avg_section_length": (
        "How long is a typical section, i.e. the text between headings or topic changes?",
        {
            "short": "Sections are brief: a few sentences, under about 150 words.",
            "medium": "Sections are a few paragraphs, about 150 to 500 words.",
            "long": "Sections are long, over about 500 words, or run on for pages.",
        },
    ),
    "profile_domain": (
        "Which subject domain does this document belong to?",
        {
            "legal": "Contracts, laws, regulations, court material, legal analysis.",
            "financial": "Finance, accounting, investing, banking, markets, financial reporting.",
            "technical": "Software, engineering, IT, science and technical documentation.",
            "general": "General interest, business, news or other content fitting no specific domain.",
            "medical": "Medicine, health care, clinical or biomedical material.",
            "policy": "Government or organisational policy, governance, standards and compliance rules.",
        },
    ),
}

# Metadata choice fields. `metadata_domain` is restricted to EXACTLY these 5
# values (they feed /api/sources/facets and frontend filters).
METADATA_CHOICES: dict[str, tuple[str, dict[str, str]]] = {
    "kind": (
        "What kind of document is this?",
        {
            "transcript": "A transcript of spoken content such as a talk, interview, podcast or meeting.",
            "paper": "An academic or research paper.",
            "article": "A news, blog or magazine article.",
            "report": "A report, study, whitepaper or analysis.",
            "infographic": "An infographic or mostly visual summary with short text fragments.",
            "playbook": "A playbook or step-by-step operating procedure for a team.",
            "guide": "A guide, tutorial, how-to or handbook.",
            "policy": "A policy, terms, regulation or formal rules document.",
            "specification": "A technical or product specification, requirements or API reference.",
            "other": "None of the other kinds fits.",
        },
    ),
    "metadata_domain": (
        "Which subject domain does this document belong to?",
        {
            "legal": "Contracts, laws, regulations, court material, legal analysis.",
            "financial": "Finance, accounting, investing, banking, markets, financial reporting.",
            "technical": "Software, engineering, IT, science and technical documentation.",
            "medical": "Medicine, health care, clinical or biomedical material.",
            "general": "General interest, business, news or other content fitting no specific domain.",
        },
    ),
}

METADATA_DOMAINS = tuple(METADATA_CHOICES["metadata_domain"][1])
KINDS = tuple(METADATA_CHOICES["kind"][1])

_PROFILE_VALUES = {name: tuple(spec[1]) for name, spec in PROFILE_CHOICES.items()}


class JevProfilingError(RuntimeError):
    """The Jev profiling call returned unusable answers."""


@dataclass
class JevClassification:
    profile: DocumentProfile
    kind: str | None = None
    metadata_domain: str | None = None
    # field -> averaged probability of the winning option (for diagnostics)
    confidence: dict[str, float] = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deterministic features
# ---------------------------------------------------------------------------


def _is_fence(line: str) -> bool:
    stripped = line.lstrip(" ")
    return len(line) - len(stripped) <= 3 and stripped.startswith(("```", "~~~"))


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    return (
        "|" in stripped
        and "-" in stripped
        and set(stripped) <= set("|-: \t")
    )


def detect_tables_and_code(markdown: str) -> tuple[bool, bool]:
    """Return `(has_tables, has_code_blocks)` from the markdown text.

    A table is a markdown table separator row (`|---|:---:|`); separator-like
    lines inside fenced code blocks do not count. A code block is any fenced
    block (``` or ~~~).
    """
    has_tables = False
    has_code = False
    in_fence = False
    for line in markdown.splitlines():
        if _is_fence(line):
            has_code = True
            in_fence = not in_fence
            continue
        if not in_fence and not has_tables and _is_table_separator(line):
            has_tables = True
        if has_tables and has_code:
            break
    return has_tables, has_code


# ---------------------------------------------------------------------------
# Question building and answer aggregation
# ---------------------------------------------------------------------------


def _ab_keys(name: str) -> tuple[str, str]:
    return f"{name}__a", f"{name}__b"


def _choice_questions(
    choices: dict[str, tuple[str, dict[str, str]]],
) -> dict[str, dict]:
    questions: dict[str, dict] = {}
    for name, (instructions, options) in choices.items():
        key_a, key_b = _ab_keys(name)
        questions[key_a] = jev.choice(instructions, options)
        questions[key_b] = jev.choice(instructions, dict(reversed(list(options.items()))))
    return questions


def build_questions(include_metadata: bool = True) -> dict[str, dict]:
    """All Jev questions: each Choice twice (options in reversed order the
    second time). `include_metadata=False` asks only the profile fields."""
    questions = _choice_questions(PROFILE_CHOICES)
    if include_metadata:
        questions.update(_choice_questions(METADATA_CHOICES))
    return questions


def average_choice(
    answers: dict[str, dict],
    name: str,
    options: tuple[str, ...] | list[str],
) -> tuple[str, float]:
    """Average the `__a`/`__b` probability distributions for `name`.

    Returns `(winning_option, averaged_probability)`. Probabilities for keys
    that are not allowed options are ignored; an answer without probabilities
    counts as full probability on its chosen key. Ties resolve to the first
    option in canonical order. Raises `JevProfilingError` when neither answer
    is usable.
    """
    totals = {option: 0.0 for option in options}
    used = 0
    for key in _ab_keys(name):
        answer = answers.get(key)
        if not isinstance(answer, dict):
            continue
        try:
            probs = jev.choice_probs(answer)
            if not probs:
                chosen = jev.choice_of(answer)
                probs = {chosen: 1.0}
        except (KeyError, TypeError, ValueError):
            continue
        valid = {option: probs[option] for option in options if option in probs}
        if not valid:
            continue
        for option, probability in valid.items():
            totals[option] += probability
        used += 1
    if not used:
        raise JevProfilingError(f"Jev returned no usable answer for '{name}'")
    best = max(options, key=lambda option: totals[option])
    return best, totals[best] / used


def classify_from_answers(
    answers: dict[str, dict],
    markdown: str,
    *,
    include_metadata: bool = True,
    usage: dict | None = None,
) -> JevClassification:
    """Turn raw Jev answers into a `JevClassification`.

    `has_tables`/`has_code_blocks` come from the full `markdown`.
    """
    confidence: dict[str, float] = {}
    values: dict[str, str] = {}
    for name, options in _PROFILE_VALUES.items():
        values[name], confidence[name] = average_choice(answers, name, options)
    has_tables, has_code_blocks = detect_tables_and_code(markdown)
    profile = DocumentProfile(
        structure_type=values["structure_type"],
        heading_consistency=values["heading_consistency"],
        content_density=values["content_density"],
        primary_content_type=values["primary_content_type"],
        avg_section_length=values["avg_section_length"],
        has_tables=has_tables,
        has_code_blocks=has_code_blocks,
        domain=values["profile_domain"],
    )
    kind = metadata_domain = None
    if include_metadata:
        kind, confidence["kind"] = average_choice(answers, "kind", KINDS)
        metadata_domain, confidence["metadata_domain"] = average_choice(
            answers, "metadata_domain", METADATA_DOMAINS
        )
    return JevClassification(
        profile=profile,
        kind=kind,
        metadata_domain=metadata_domain,
        confidence={name: round(value, 4) for name, value in confidence.items()},
        usage=usage or {},
    )


def jev_classify(markdown: str, *, include_metadata: bool = True) -> JevClassification:
    """ONE Jev call classifying the profile (and optionally kind/domain).

    Raises on any Jev failure (including a missing API key); callers fall back
    to the LLM path.
    """
    if not markdown or not markdown.strip():
        raise JevProfilingError("empty document")
    state = {"document_excerpt": markdown[:EXCERPT_CHARS]}
    response = jev.system_one_response(state, build_questions(include_metadata))
    return classify_from_answers(
        response.answers, markdown, include_metadata=include_metadata, usage=response.usage
    )


def extract_profile_and_metadata(markdown: str) -> tuple[dict, JevClassification]:
    """Parsing-stage helper: Jev profile + kind/domain and LLM author/source,
    run concurrently. Returns `(metadata, classification)` where `metadata`
    has the same keys as `extract_metadata` (kind, author, source, domain).

    Raises when the Jev call fails (the author/source call never raises); the
    caller then falls back to the full legacy `extract_metadata`.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        jev_future = pool.submit(jev_classify, markdown, include_metadata=True)
        author_future = pool.submit(extract_author_source, markdown)
        author_source = author_future.result()
        classification = jev_future.result()
    metadata: dict[str, Any] = {
        "kind": classification.kind,
        "domain": classification.metadata_domain,
    }
    metadata.update(author_source)
    return metadata, classification


# ---------------------------------------------------------------------------
# Persistence helpers (parsing stage_log output -> profiling stage)
# ---------------------------------------------------------------------------


def profile_to_dict(profile: DocumentProfile) -> dict:
    return dataclasses.asdict(profile)


def profile_from_dict(data: Any) -> DocumentProfile | None:
    """Rebuild a persisted profile; `None` when absent or malformed."""
    if not isinstance(data, dict):
        return None
    try:
        for name, allowed in _PROFILE_VALUES.items():
            target = "domain" if name == "profile_domain" else name
            if data.get(target) not in allowed:
                return None
        if not isinstance(data.get("has_tables"), bool) or not isinstance(
            data.get("has_code_blocks"), bool
        ):
            return None
        return DocumentProfile(**{f.name: data[f.name] for f in dataclasses.fields(DocumentProfile)})
    except (KeyError, TypeError):
        return None
