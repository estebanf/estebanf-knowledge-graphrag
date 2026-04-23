import json
import re
from dataclasses import dataclass, field

import requests
import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag.config import settings
from rag.profiling import DocumentProfile

_enc = tiktoken.get_encoding("cl100k_base")

_CHUNK_SIZE = 400
_CHUNK_OVERLAP = 80
_PARENT_SIZE = 2000
_CHAR_RATIO = 4


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class ChunkData:
    content: str
    token_count: int
    chunk_index: int
    parent_chunk_id: str | None
    chunking_strategy: str
    chunking_config: dict
    metadata: dict = field(default_factory=dict)


def select_strategy(profile: DocumentProfile) -> str:
    if (profile.structure_type == "well-structured"
            and profile.heading_consistency == "consistent"):
        return "markdown-header"
    if profile.primary_content_type in {"transcript", "qa_pairs"}:
        return "semantic"
    if profile.content_density == "uniform" and profile.structure_type == "unstructured":
        return "recursive"
    return "semantic"


def _recursive_splitter(chunk_size_tokens: int = _CHUNK_SIZE) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens * _CHAR_RATIO,
        chunk_overlap=_CHUNK_OVERLAP * _CHAR_RATIO,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _is_heading_only(text: str) -> bool:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return bool(lines) and all(re.match(r"^#{1,6}\s+", line) for line in lines)


def _normalize_markdown_chunk(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _split_markdown_header(text: str) -> list[str]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    docs = splitter.split_text(text)
    secondary = _recursive_splitter()
    result = []
    for doc in docs:
        if _token_count(doc.page_content) > _CHUNK_SIZE:
            result.extend(secondary.split_text(doc.page_content))
        else:
            result.append(doc.page_content)
    result = [_normalize_markdown_chunk(t) for t in result if t.strip()]

    merged: list[str] = []
    pending_prefix: list[str] = []
    for chunk in result:
        if _is_heading_only(chunk):
            pending_prefix.append(chunk)
            continue
        if pending_prefix:
            merged.append("\n\n".join([*pending_prefix, chunk]).strip())
            pending_prefix = []
        else:
            merged.append(chunk)
    if pending_prefix:
        merged.append("\n\n".join(pending_prefix).strip())
    return merged


def _split_recursive(text: str, chunk_size_tokens: int = _CHUNK_SIZE) -> list[str]:
    splitter = _recursive_splitter(chunk_size_tokens)
    return [t for t in splitter.split_text(text) if t.strip()]


def _split_semantic(text: str, chunk_size_tokens: int = _CHUNK_SIZE) -> list[str]:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if part.strip()
    ]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _token_count(sentence)
        if current and current_tokens + sentence_tokens > chunk_size_tokens:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def _split_by_strategy(text: str, strategy: str, chunk_size_tokens: int = _CHUNK_SIZE) -> list[str]:
    if strategy == "markdown-header":
        return _split_markdown_header(text)
    if strategy == "semantic":
        return _split_semantic(text, chunk_size_tokens)
    return _split_recursive(text, chunk_size_tokens)


def _decompose_to_propositions(text: str) -> list[str]:
    if not settings.OPENROUTER_API_KEY:
        return []
    prompt = (
        "Break the following text into atomic, self-contained propositions. "
        "Each must be a single factual statement understandable without context.\n"
        "Return ONLY a JSON array of strings.\n\nText:\n" + text
    )
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.MODEL_PROPOSITION_CHUNKING,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1024,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return json.loads(content)
    except Exception:
        return []


def chunk_document(markdown: str, profile: DocumentProfile) -> list[ChunkData]:
    strategy = select_strategy(profile)
    use_hierarchical = profile.avg_section_length == "long"
    use_propositions = (
        profile.domain in ("legal", "financial", "medical")
        and bool(settings.OPENROUTER_API_KEY)
    )

    base_config = {"chunk_size_tokens": _CHUNK_SIZE, "chunk_overlap_tokens": _CHUNK_OVERLAP}

    # Base splitting
    if use_hierarchical:
        parent_texts = _split_by_strategy(markdown, strategy, _PARENT_SIZE)
        raw_items: list[tuple[str, str, int | None]] = []
        running_idx = 0
        for parent_text in parent_texts:
            parent_idx = running_idx
            raw_items.append((parent_text, strategy, None))
            running_idx += 1
            children = _split_by_strategy(parent_text, strategy, _CHUNK_SIZE)
            for child in children:
                raw_items.append((child, "hierarchical", parent_idx))
                running_idx += 1
    else:
        texts = _split_by_strategy(markdown, strategy)
        raw_items = [(t, strategy, None) for t in texts]

    # Build initial chunks
    chunks: list[ChunkData] = []
    for i, (text, strat, parent_idx_ref) in enumerate(raw_items):
        chunks.append(ChunkData(
            content=text,
            token_count=_token_count(text),
            chunk_index=i,
            parent_chunk_id=str(parent_idx_ref) if parent_idx_ref is not None else None,
            chunking_strategy=strat,
            chunking_config=base_config,
            metadata={"base_strategy": strategy} if parent_idx_ref is not None else {},
        ))

    if not use_propositions:
        return chunks

    # Proposition expansion
    final_chunks: list[ChunkData] = []
    for chunk in chunks:
        propositions = _decompose_to_propositions(chunk.content)
        if not propositions:
            final_chunks.append(chunk)
            continue
        parent_idx = len(final_chunks)
        final_chunks.append(ChunkData(
            content=chunk.content,
            token_count=chunk.token_count,
            chunk_index=parent_idx,
            parent_chunk_id=None,
            chunking_strategy=chunk.chunking_strategy,
            chunking_config=chunk.chunking_config,
        ))
        for prop in propositions:
            final_chunks.append(ChunkData(
                content=prop,
                token_count=_token_count(prop),
                chunk_index=len(final_chunks),
                parent_chunk_id=str(parent_idx),
                chunking_strategy="proposition",
                chunking_config={"parent_strategy": chunk.chunking_strategy},
                metadata={"parent_chunk_index": parent_idx},
            ))

    # Reindex
    for i, c in enumerate(final_chunks):
        c.chunk_index = i
    return final_chunks
