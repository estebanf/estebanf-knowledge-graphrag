# Prompt Extraction Design

**Date:** 2026-04-25

## Goal

Move all LLM prompt strings out of individual Python modules and into a single `src/rag/prompts/__init__.py` module. This makes prompts easy to find and edit without touching application logic.

## Approach

Create `src/rag/prompts/__init__.py` with one named string constant per prompt. Each source module imports from `rag.prompts` and removes its local prompt definition.

For prompts that have runtime values injected (query text, entity names, chunk content), the constant holds only the static instruction text. The call site continues to concatenate or `.format()` at runtime — no change to that logic.

## Prompt Constants

| Constant | Source file | Notes |
|---|---|---|
| `DOCUMENT_PROFILING` | `profiling.py` | Static; appended with document excerpt at call time |
| `CHUNK_VALIDATION` | `chunk_validation.py` | Static; appended with chunk content at call time |
| `ENTITY_EXTRACTION` | `graph_extraction.py` | Template with `{types}` and `{text}` — `.format()` unchanged |
| `RELATIONSHIP_EXTRACTION` | `graph_extraction.py` | Template with `{entity_names}` and `{text}` — `.format()` unchanged |
| `PROPOSITION_DECOMPOSITION` | `chunking.py` | Static; concatenated with text at call time |
| `ANSWER_GENERATION` | `answering.py` | Static instruction block; `_build_answer_prompt` wraps it with query and JSON |
| `COMMUNITY_SUMMARIZATION` | `community.py` | Default fallback value (overridable via `settings.COMMUNITY_SUMMARIZATION_PROMPT`) |
| `QUERY_VARIANTS` | `retrieval.py` | Template with `{max_decomposed}` and `{query}` |
| `ENTITY_SELECTION` | `retrieval.py` | Template with `{max_entities}`, `{query}`, `{seed_chunk}`, `{entities}` |
| `ENTITY_QUERY_GENERATION` | `retrieval.py` | Template with `{query}`, `{seed_chunk}`, `{entity_name}` |
| `SECOND_HOP_ENTITY_SELECTION` | `retrieval.py` | Template with `{max_entities}`, `{query}`, `{seed_chunk}`, `{entity_name}`, `{candidates}` |

## What Does Not Change

- Call sites: concatenation, `.format()`, and f-string assembly logic stays in each module.
- LLM invocation logic, model selection, error handling — all unchanged.
- Tests: no prompts are directly tested; behavior tests remain valid.
