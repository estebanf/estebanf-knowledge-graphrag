# Prompt Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move all LLM prompt strings from individual Python modules into `src/rag/prompts/__init__.py` so they can be found and edited in one place.

**Architecture:** A new `rag.prompts` module holds one named string constant per prompt. Modules that currently define prompt strings drop their local constant and import from `rag.prompts`. For prompts with runtime values (query text, entity lists), the constant becomes a `.format()`-style template and the call site passes named arguments.

**Tech Stack:** Pure Python, no new dependencies.

---

### Task 1: Create `rag/prompts/__init__.py` with all constants

**Files:**
- Create: `src/rag/prompts/__init__.py`

**Step 1: Write a failing smoke test**

Add to `tests/test_prompts.py` (create file):

```python
from rag import prompts


def test_all_prompt_constants_are_non_empty_strings():
    constants = [
        prompts.DOCUMENT_PROFILING,
        prompts.CHUNK_VALIDATION,
        prompts.ENTITY_EXTRACTION,
        prompts.RELATIONSHIP_EXTRACTION,
        prompts.PROPOSITION_DECOMPOSITION,
        prompts.ANSWER_GENERATION,
        prompts.COMMUNITY_SUMMARIZATION,
        prompts.QUERY_VARIANTS,
        prompts.ENTITY_SELECTION,
        prompts.ENTITY_QUERY_GENERATION,
        prompts.SECOND_HOP_ENTITY_SELECTION,
    ]
    for c in constants:
        assert isinstance(c, str) and c.strip(), f"Expected non-empty string, got {c!r}"
```

**Step 2: Run to verify it fails**

```bash
cd /path/to/repo
pytest tests/test_prompts.py -v
```

Expected: `ImportError` — module does not exist yet.

**Step 3: Create `src/rag/prompts/__init__.py`**

```python
DOCUMENT_PROFILING = """\
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

CHUNK_VALIDATION = """\
Evaluate this document chunk for retrieval quality. Score it on:
- completeness: does it contain complete thoughts?
- coherence: is it internally coherent?
- appropriate_length: is it neither too short nor too long?

Return ONLY: {"pass": true} or {"pass": false, "reason": "brief explanation"}

Chunk:
"""

ENTITY_EXTRACTION = """Extract named entities from the following text. Return ONLY a JSON array.
Each item must have: "canonical_name" (string), "entity_type" (one of: {types}), "aliases" (list of strings).
Return [] if no entities found.

Text:
{text}"""

RELATIONSHIP_EXTRACTION = """Given these entities: {entity_names}

Extract relationships from the text. Return ONLY a JSON array.
Each item must have: "source" (canonical_name), "target" (canonical_name), "type" (string), "confidence" (0.0-1.0).
Use concise relationship types like OWNS, EMPLOYS, GOVERNS, REQUIRES, IS_PART_OF, RELATED_TO.
Only include relationships grounded in the text.
Return [] if none found.

Text:
{text}"""

PROPOSITION_DECOMPOSITION = (
    "Break the following text into atomic, self-contained propositions. "
    "Each must be a single factual statement understandable without context.\n"
    "Return ONLY a JSON array of strings.\n\nText:\n"
)

ANSWER_GENERATION = (
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
)

COMMUNITY_SUMMARIZATION = "Craft a compelling narrative summarizing these chunks of related information"

QUERY_VARIANTS = """\
You are generating bounded retrieval query variants.
Return ONLY a JSON object with keys: original, hyde, expanded, step_back, decomposed.
- original must be the exact input query
- hyde must be a short hypothetical answer passage for dense retrieval
- expanded must add synonyms, aliases, abbreviations, and related terms
- step_back must be a more general background query
- decomposed must contain at most {max_decomposed} focused sub-queries
- avoid near-duplicate variants

Query:
{query}
"""

ENTITY_SELECTION = """\
You are selecting graph entities to expand for retrieval.
Return ONLY a JSON object with key selected_entities containing up to {max_entities} entity names.

User query:
{query}

Seed chunk:
{seed_chunk}

Entities:
{entities}
"""

ENTITY_QUERY_GENERATION = """\
You are generating a retrieval sub-query.
Return ONLY a JSON object with key query.

Original user query:
{query}

Seed chunk:
{seed_chunk}

Entity:
{entity_name}
"""

SECOND_HOP_ENTITY_SELECTION = """\
You are selecting second-hop graph entities to expand for retrieval.
Return ONLY a JSON object with key selected_entities containing up to {max_entities} entity names.

User query:
{query}

Seed chunk:
{seed_chunk}

Current entity:
{entity_name}

Candidates:
{candidates}
"""
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_prompts.py -v
```

Expected: PASS — all 11 constants are non-empty strings.

**Step 5: Commit**

```bash
git add tests/test_prompts.py src/rag/prompts/__init__.py
git commit -m "feat: add rag.prompts module with all LLM prompt constants"
```

---

### Task 2: Update `profiling.py`

**Files:**
- Modify: `src/rag/profiling.py:11-27`

**Step 1: Replace the local `_PROMPT` constant and its usage**

In `src/rag/profiling.py`:
- Remove lines 11–27 (the `_PROMPT = """\..."""` block)
- Add `from rag import prompts` to the imports
- Replace `_PROMPT + sample` with `prompts.DOCUMENT_PROFILING + sample`

After editing, lines 1–10 of the file should look like:

```python
import json
import re
from dataclasses import dataclass

import requests

from rag import prompts
from rag.config import settings
```

And the payload in `profile_document`:
```python
"messages": [{"role": "user", "content": prompts.DOCUMENT_PROFILING + sample}],
```

**Step 2: Run the existing test suite**

```bash
pytest tests/ -v -k "profil"
```

Expected: same pass/fail count as before.

**Step 3: Commit**

```bash
git add src/rag/profiling.py
git commit -m "refactor: move document profiling prompt to rag.prompts"
```

---

### Task 3: Update `chunk_validation.py`

**Files:**
- Modify: `src/rag/chunk_validation.py:13-22`

**Step 1: Replace the local `_PROMPT` constant**

In `src/rag/chunk_validation.py`:
- Remove lines 13–22 (the `_PROMPT = """\..."""` block)
- Add `from rag import prompts` to the imports
- Replace `_PROMPT + content` with `prompts.CHUNK_VALIDATION + content`

Imports section after edit:
```python
import json
import math
import random
import re

import requests

from rag import prompts
from rag.chunking import ChunkData
from rag.config import settings
```

Payload line:
```python
"messages": [{"role": "user", "content": prompts.CHUNK_VALIDATION + content}],
```

**Step 2: Run tests**

```bash
pytest tests/ -v -k "validat"
```

Expected: same result as before.

**Step 3: Commit**

```bash
git add src/rag/chunk_validation.py
git commit -m "refactor: move chunk validation prompt to rag.prompts"
```

---

### Task 4: Update `graph_extraction.py`

**Files:**
- Modify: `src/rag/graph_extraction.py:12-28`

**Step 1: Replace both local prompt constants**

In `src/rag/graph_extraction.py`:
- Remove lines 12–28 (`_ENTITY_PROMPT` and `_REL_PROMPT`)
- Add `from rag import prompts` to imports
- Replace `_ENTITY_PROMPT.format(...)` with `prompts.ENTITY_EXTRACTION.format(...)`
- Replace `_REL_PROMPT.format(...)` with `prompts.RELATIONSHIP_EXTRACTION.format(...)`

The `.format()` call signatures do not change — only the constant name changes.

**Step 2: Run tests**

```bash
pytest tests/ -v -k "graph_extract or entity"
```

Expected: same result as before.

**Step 3: Commit**

```bash
git add src/rag/graph_extraction.py
git commit -m "refactor: move graph extraction prompts to rag.prompts"
```

---

### Task 5: Update `chunking.py`

**Files:**
- Modify: `src/rag/chunking.py:139-143`

**Step 1: Replace the inline prompt in `_decompose_to_propositions`**

In `src/rag/chunking.py`, the function `_decompose_to_propositions` currently builds `prompt` by concatenating a long string literal with `text`. Replace that with:

```python
from rag import prompts  # add to imports at top of file

# inside _decompose_to_propositions:
prompt = prompts.PROPOSITION_DECOMPOSITION + text
```

The existing `requests.post(...)` call that uses `prompt` is unchanged.

**Step 2: Run tests**

```bash
pytest tests/ -v -k "chunk"
```

Expected: same result as before.

**Step 3: Commit**

```bash
git add src/rag/chunking.py
git commit -m "refactor: move proposition decomposition prompt to rag.prompts"
```

---

### Task 6: Update `answering.py`

**Files:**
- Modify: `src/rag/answering.py:26-55`

**Step 1: Rewrite `_build_answer_prompt`**

The current function builds the full string inline. After the refactor it assembles from the constant:

```python
from rag import prompts  # add to imports at top of file

def _build_answer_prompt(query: str, results: dict) -> str:
    return (
        f'User question: "{query}"\n\n'
        + prompts.ANSWER_GENERATION
        + f"```\n{json.dumps(results, indent=2)}\n```"
    )
```

Note: `ANSWER_GENERATION` already ends with `"Retrieved results:\n"`, so the f-string above appends the code-fenced JSON immediately after.

**Step 2: Run tests**

```bash
pytest tests/ -v -k "answer"
```

Expected: same result as before.

**Step 3: Commit**

```bash
git add src/rag/answering.py
git commit -m "refactor: move answer generation prompt to rag.prompts"
```

---

### Task 7: Update `community.py`

**Files:**
- Modify: `src/rag/community.py:366-368`

**Step 1: Replace the inline fallback string**

Current code:
```python
prompt = settings.COMMUNITY_SUMMARIZATION_PROMPT or (
    "Craft a compelling narrative summarizing these chunks of related information"
)
```

After:
```python
from rag import prompts  # add to imports at top of file

prompt = settings.COMMUNITY_SUMMARIZATION_PROMPT or prompts.COMMUNITY_SUMMARIZATION
```

**Step 2: Run tests**

```bash
pytest tests/ -v -k "communit"
```

Expected: same result as before.

**Step 3: Commit**

```bash
git add src/rag/community.py
git commit -m "refactor: move community summarization prompt to rag.prompts"
```

---

### Task 8: Update `retrieval.py`

**Files:**
- Modify: `src/rag/retrieval.py` — four functions: `generate_query_variants` (line 205), `_select_entities` (line 527), `_generate_entity_query` (line 556), `_select_second_hop_entities` (line 777)

**Step 1: Add import**

At the top of `src/rag/retrieval.py`, add:
```python
from rag import prompts
```

**Step 2: Rewrite `generate_query_variants` prompt (line 205)**

Replace the f-string block with:
```python
prompt = prompts.QUERY_VARIANTS.format(
    max_decomposed=settings.RETRIEVAL_MAX_DECOMPOSED_QUERIES,
    query=query,
)
```

**Step 3: Rewrite `_select_entities` prompt (line 527)**

Replace the f-string block with:
```python
prompt = prompts.ENTITY_SELECTION.format(
    max_entities=settings.RETRIEVAL_ENTITY_SELECTION_COUNT,
    query=query,
    seed_chunk=seed.chunk[:1500],
    entities=json.dumps([entity.__dict__ for entity in entities], ensure_ascii=True),
)
```

**Step 4: Rewrite `_generate_entity_query` prompt (line 556)**

Replace the f-string block and the conditional append with:
```python
prompt = prompts.ENTITY_QUERY_GENERATION.format(
    query=query,
    seed_chunk=seed.chunk[:1200],
    entity_name=entity_name,
)
if entity_context:
    prompt += f"\nPath context:\n{json.dumps(entity_context, ensure_ascii=True)}\n"
```

**Step 5: Rewrite `_select_second_hop_entities` prompt (line 777)**

Replace the f-string block with:
```python
prompt = prompts.SECOND_HOP_ENTITY_SELECTION.format(
    max_entities=settings.RETRIEVAL_SECOND_HOP_SELECTION_COUNT,
    query=query,
    seed_chunk=seed.chunk[:1200],
    entity_name=entity_name,
    candidates=json.dumps([candidate.__dict__ for candidate in deduped_candidates], ensure_ascii=True),
)
```

**Step 6: Run tests**

```bash
pytest tests/ -v -k "retriev"
```

Expected: same result as before.

**Step 7: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all 85 tests pass.

**Step 8: Commit**

```bash
git add src/rag/retrieval.py
git commit -m "refactor: move retrieval prompts to rag.prompts"
```

---

### Task 9: Final verification

**Step 1: Confirm no stray prompt strings remain in source modules**

```bash
grep -rn "_PROMPT\s*=" src/rag/ --include="*.py"
```

Expected: no output.

**Step 2: Confirm all prompts module constants are importable**

```bash
python -c "from rag import prompts; print('OK')"
```

Expected: `OK`

**Step 3: Run full test suite one final time**

```bash
pytest tests/ -v
```

Expected: 85 tests pass, 0 failures.
