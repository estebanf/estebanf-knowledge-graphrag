# Answer Prompt Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current answer prompt with a relevance-first structured prompt that produces clean narrative answers and suppresses tangential retrieval themes.

**Architecture:** Keep the change local to `src/rag/answering.py` by updating `_build_answer_prompt()`. Lock the new behavior with a focused regression test in `tests/test_answering.py`, then verify it against the backend API suite.

**Tech Stack:** Python, pytest

### Task 1: Tighten the prompt contract in tests

**Files:**
- Modify: `tests/test_answering.py`
- Test: `tests/test_answering.py`

**Step 1: Write the failing test**

Update the prompt assertions to require:

- "Answer the user's exact question as directly as possible."
- "First identify the 2 to 4 chunks that are most directly relevant to the question."
- "Ignore interesting but tangential themes, even if they are high scoring."
- "Do not organize the answer around the structure of the retrieval."
- "Do not mention chunk IDs, scores, or retrieval mechanics in the final answer."

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_answering.py`

Expected: `FAIL` because the current prompt does not include the stricter instructions.

### Task 2: Update the answer prompt

**Files:**
- Modify: `src/rag/answering.py`
- Test: `tests/test_answering.py`

**Step 1: Write minimal implementation**

Update `_build_answer_prompt()` to:

- make the question-first task explicit,
- add an internal evidence-selection step,
- prioritize narrow domain cues,
- suppress tangential high-scoring material,
- require the fixed narrative structure,
- forbid retrieval identifiers and retrieval-language in the final answer.

**Step 2: Run test to verify it passes**

Run: `pytest -q tests/test_answering.py`

Expected: `PASS`

### Task 3: Verify no backend regression

**Files:**
- Test: `tests/test_answering.py`
- Test: `tests/test_api.py`

**Step 1: Run backend verification**

Run: `pytest -q tests/test_answering.py tests/test_api.py`

Expected: `PASS`
