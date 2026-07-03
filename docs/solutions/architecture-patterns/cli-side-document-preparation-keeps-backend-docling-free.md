---
title: "CLI-Side Document Preparation Keeps the Backend Docling-Free"
date: 2026-07-03
category: architecture-patterns
module: ingestion
problem_type: architecture_pattern
component: tooling
severity: high
applies_when:
  - "A shared package installs a heavy optional dependency (ML/document-parsing library pulling in Torch/CUDA-class transitive deps) via an editable extra that only one deployment target actually needs"
  - "Deciding which side of a CLI/backend split should own a capability when only the CLI-invoked path needs the heavy dependency"
  - "A container's rebuild size/time has grown large enough to risk disk exhaustion or slow iteration, and the growth traces to an optional extra rather than core runtime code"
  - "Refactoring so a component being dependency-free is enforced by a test that imports it in a fresh process and asserts the dependency never appears in sys.modules, rather than relying on code review alone"
symptoms:
  - "Backend Docker image ballooned to multiple GB and took a long time to rebuild because `.[ingest]` pulled in docling>=2.0 and its Torch/torchvision/transformers/accelerate/NVIDIA CUDA transitive dependencies"
  - "A backend image rebuild filled the host disk to 100% and corrupted the local Docker overlay2 filesystem, a real operational incident"
  - "No automated guardrail existed to catch a regression where docling-related imports crept back into backend-only modules"
resolution_type: code_fix
related_components: [cli, backend-api, worker, docker, ingestion-pipeline, parser]
tags: [docling, docker-image-size, cli-backend-boundary, prepared-binary-ingestion, dependency-isolation, pyproject-extras, ingestion-pipeline]
---

# CLI-Side Document Preparation Keeps the Backend Docling-Free

## Context

The backend Docker image for this RAG system had quietly become enormous — 6.04GB — because `pyproject.toml` installed the `.[ingest]` optional extra directly into the backend's container image, and that extra pulled in `docling>=2.0` for parsing PDF/DOCX/PPTX documents. Docling's dependency tree is not small: it drags in Torch, torchvision, transformers, accelerate, and a set of NVIDIA CUDA wheels the backend has no actual use for, since the backend's real job is serving API requests, running the ingestion worker, and talking to Postgres/Memgraph — not running local ML inference for document layout parsing.

This had already been flagged once, in `docs/solutions/performance-issues/rag-retrieval-vector-prefilter-and-query-fanout.md`, as a source of slow, multi-gigabyte rebuilds. The cost of ignoring it stopped being theoretical during this same working session: a backend rebuild pulling the full Torch/CUDA stack filled the host disk to 100% and corrupted the local Docker overlay2 filesystem. That was a real infrastructure incident, not a hypothetical one, and it's the concrete trigger for treating "the backend links a GPU-scale ML stack for a task it doesn't need to do itself" as an architectural defect worth fixing at the design level rather than patching around (e.g. by just enlarging disk or trimming caches).

A Codex-authored plan (`docs/plans/2026-07-03-001-refactor-prepared-binary-ingestion-plan.md`, units U1–U8) proposed the structural fix: stop parsing binary documents inside the backend at all. Move that work to wherever the user already has the file — the CLI, running on their own machine — and have the CLI submit already-converted markdown to the backend. The backend keeps a single narrow, scoped endpoint for the one piece of Docling's job it can't safely delegate to the client: describing extracted images via the LLM, since that requires the `OPENROUTER_API_KEY` and image-model config that must stay server-side.

## Guidance

The core pattern applied here is **CLI-side heavy-dependency isolation via a prepare/parse split**, plus a small set of supporting decisions that make the split trustworthy rather than just aspirational.

**1. Split "prepare" (binary → markdown, needs Docling) from "parse" (markdown/text, never needs Docling), and make the split structural, not conditional.**

`src/rag/parser.py` used to handle both text formats and binary formats behind Docling. Now it handles only markdown/text, and raises immediately (`ParseError`) for any `.pdf`/`.docx`/`.pptx` suffix — without ever importing Docling to make that decision. All Docling logic moved to a new module, `src/rag/prepare.py`:

```python
# src/rag/prepare.py
def prepare_binary(path: Path) -> PreparedDocument:
    """Convert a binary document (PDF/DOCX/PPTX) to markdown + extracted
    images, using Docling. Returns markdown text, a list of extracted
    image bytes in placeholder order, and original file provenance
    (md5, filename, extension)."""
    ...

def finalize_markdown(prepared: PreparedDocument, describe_fn) -> str:
    """Replace Docling's `<!-- image -->` placeholders, in order, with
    descriptions produced by describe_fn for each extracted image.
    Raises PrepareError if any placeholder remains unresolved after
    every extracted image has been described."""
    ...
```

The important property is not just "Docling lives in a different file" — it's that `parser.py` can no longer import Docling even by accident, because the binary-suffix branch is a hard raise, not a conditional call into `prepare.py`. That makes "the backend is Docling-free" true by construction rather than by convention, which matters because conventions erode under time pressure and code review can't fully substitute for a structural guarantee.

**2. Keep the one unavoidable server-side call (image captioning) behind a narrow, defensive, scoped endpoint.**

The CLI needs images described, but must never hold `OPENROUTER_API_KEY` or image-model configuration. So a new endpoint, `POST /api/prepare/describe-image`, lets the CLI hand the backend raw image bytes and get a caption back, with the backend keeping full custody of credentials and config. It:
- validates the MIME type against an allow-list,
- validates the base64 payload,
- caps the payload at 10MB *before* decoding (to bound memory blowup from a malicious or malformed request),
- and masks any upstream LLM failure behind a generic 502, never leaking upstream error text or config details to the client.

Wiring this endpoint up surfaced a latent bug in `src/rag/api/auth.py`: `requires_scope()` built a brand-new `require_principal()` closure on every call. That's a problem specifically because FastAPI's `app.dependency_overrides[...]` test-bypass mechanism keys on *object identity* — a route depending on a fresh closure each time would silently ignore a test's auth-bypass override, since the override is registered against a different object than the one actually resolved at request time. The fix was to have `requires_scope()` sub-depend on one shared `default_principal_dependency` singleton instead, so every route (including this new one) shares the same override-able dependency object as `main.py`'s existing `principal_dep`.

**3. Preserve duplicate-detection semantics across the prepare/submit split by keeping the hash keyed on the *original* file, not the *submitted* content.**

Before this change, `sources.md5` was implicitly "the hash of whatever was submitted." After the split, what's submitted (markdown) and what the user actually cares about deduplicating (the original PDF/DOCX/PPTX) are different byte streams. The decision was to keep `sources.md5` — and its pre-existing unique partial index — storing the hash of the *original binary file*, threaded through as explicit provenance on the text-submission path:

```python
# CLI computes original_md5/file_name/file_type from the source binary,
# then submits the *prepared* markdown text carrying that provenance:
client.submit_text(
    content=finalized_markdown,
    original_md5=original_md5,      # hash of the source PDF/DOCX/PPTX
    file_name="quarterly-report.pdf",
    file_type="pdf",
)

# Backend: submit_ingestion_job() / check_duplicate() now key off
# original_md5 for prepared submissions instead of hashing the
# markdown text, so re-submitting the same source document is still
# recognized as a duplicate even though the markdown was regenerated.
```

This required zero schema migration, but it was only safe after grepping across `src/` and `scripts/` to confirm `check_duplicate()` is the *only* reader of `sources.md5`, and that no verification/repair script independently recomputes-and-compares that column in a way this repurposing would break. The generalizable lesson: reusing an existing column's semantics for a new purpose is fine, but only after you've enumerated every reader/writer of that column, not just the one you're changing.

**4. Prefer hard-fail over silently leaking a broken invariant into stored data.**

The old parser, when an image description failed, would silently `continue`, leaving a bare `<!-- image -->` placeholder permanently embedded in the stored markdown. The new `finalize_markdown()` inverts that: if any placeholder is still unresolved after every extracted image has gone through `describe_fn`, it raises `PrepareError` and nothing gets stored. Correspondingly, `rag ingest` now aborts *before queuing any backend job* if preparation or image description fails anywhere in the pipeline — no partially-broken job ever lands in the system. This is a deliberate behavior change, not an oversight: a placeholder that "mostly worked" is worse than an ingestion that visibly failed, because the former corrupts data in a way that's invisible until a much later point (e.g. a user or an LLM downstream trusting the markdown is complete).

**5. Prove the dependency is gone with an import-guard test, not a promise.**

`tests/test_cli_imports.py` spawns a fresh Python subprocess, imports `rag.api.main`, `rag.worker`, and `rag.ingestion`, and asserts `"docling" not in sys.modules` afterward. Running this in a subprocess (rather than in-process) matters: Python caches module imports for the lifetime of the interpreter, so an in-process assertion could pass only because some earlier, unrelated test in the same session hadn't imported Docling yet — or could be poisoned by test ordering. A subprocess gives a clean-room answer to "does importing the backend's actual entry points transitively import Docling?" This is the kind of test that's cheap to write and catches an accidental Docling import creeping back into the backend *before* a slow or disk-filling build reveals it in CI or production — i.e., it converts a build-time/runtime failure mode into a fast, deterministic unit test.

> **What didn't work initially — the AE2 test-coverage regression.** All 8 units were done and the full test suite (428 tests) was green, but an advisor review caught a real gap after the fact: when the pptx parser tests were relocated from `test_parser.py` to the new `test_prepare.py` during U1, the move preserved coverage of `prepare_binary()` metadata and of `finalize_markdown()` against a hand-built `PreparedDocument`, but it silently *dropped* the one assertion the plan's acceptance criterion (AE2) actually depended on: that running **real Docling** on a real pptx fixture, then piping its **actual** output through `finalize_markdown()`, leaves zero unresolved `<!-- image -->` placeholders. Every other test by then exercised `prepare_binary`/`finalize_markdown` against mocks — the one seam where a mismatch between Docling's placeholder count and its extractable-image count would actually surface had zero end-to-end coverage, even with the suite fully green. The fix was small (add `finalize_markdown()` plus a `"<!-- image -->" not in final` assertion directly to the existing real-Docling pptx test), but the discriminating power was real: had it failed, it would have meant the new hard-fail contract was too aggressive for real Docling output — e.g. if Docling emits a placeholder for a picture it can't rasterize, `picture.get_image()` returns `None`, `prepare_binary` silently skips that image, and the placeholder count then exceeds the extracted-image count, making that document permanently un-ingestible. The generalizable lesson: **splitting a test file during a refactor can silently drop the one assertion that mattered, even while total test count and pass rate stay green** — a passing suite tells you nothing about assertions that were quietly relocated out of existence. When relocating tests during a refactor, explicitly re-derive which specific assertion each acceptance criterion depends on, rather than trusting that "the tests still pass" means "the tests still check the same thing."

A second, expected (not surprising) breakage in the same vein: two pre-existing end-to-end tests, `test_ingestion.py::test_ingest_pdf` and `test_ingest_docx`, failed immediately after U7's parser.py rewrite, because they fed a real binary file straight through `ingest_file()` → `execute_ingestion_pipeline()` → `parse_document()` — exactly the code path being removed. This wasn't a regression to chase, it was confirmation the refactor did what it intended. The fix was to rewrite both tests to call `prepare_binary()` + `finalize_markdown()` first (mirroring what the CLI now does) and then ingest the resulting markdown, adding new assertions that original-file provenance (hash/filename/extension) survives into the `sources` row.

## Why This Matters

The measured impact, verified live rather than assumed:

- **Image size: 6.04GB → 388MB.** A roughly 15x reduction, because the backend image no longer contains Torch, torchvision, transformers, accelerate, or the NVIDIA CUDA wheels at all.
- **Build time: several minutes (full Torch/CUDA reinstall) → ~38 seconds.** This isn't just a convenience — it's the difference between a rebuild being an unremarkable part of the dev loop and being an event that risks the host's disk.
- **The CUDA/Torch attack and failure surface is removed from the server entirely**, not just made smaller. Confirmed live: `import docling` raises `ModuleNotFoundError` inside the running container, while `rag.api.main`, `rag.worker`, `rag.retrieval`, `rag.ingestion`, `rag.community`, and `rag.parser` all still import and run cleanly, and API health/search/retrieve endpoints work end-to-end.
- **The disk-full-during-rebuild incident becomes structurally impossible going forward**, not just less likely. The failure mode required a multi-gigabyte dependency reinstall on a backend rebuild; once that dependency cannot appear in the backend's dependency graph (proven by the subprocess import-guard test, not merely by removing a line from `pyproject.toml`), the precondition for the incident no longer exists.
- A full live smoke test validated the change didn't regress ingestion correctness: Postgres baseline (sources=7518, jobs=7518, chunks=125914, entities=113835) and Memgraph node count (360228) were captured before the smoke, a small doc was ingested through a real worker to `completed` across all 8 pipeline stages, and after hard-deleting the smoke source both Postgres and Memgraph counts returned exactly to baseline — confirming the new prepare/submit path is correct and leaves no residue, not just that it "runs."

Beyond the numbers, the broader lesson is about where dependency weight belongs: a server process should carry only the dependencies its own runtime job requires, not the dependencies of a client-triggerable *capability* that could just as easily execute where the client already is.

## When to Apply

This pattern generalizes whenever a server process links a heavy, disproportionate dependency (GPU frameworks, native rendering libraries, large ML runtimes) because of a single code path that *could* run client-side instead:

- The dependency is needed for one specific transformation (parsing, rendering, transcoding, local inference) that is triggered by, and could be executed by, the client submitting the file — not by ongoing server-side business logic.
- The dependency's install/runtime footprint (image size, build time, GPU/native library requirements) is disproportionate to what the server otherwise needs to do its actual job (serve requests, run a worker loop, talk to a database).
- There's a narrow slice of the capability that genuinely must stay server-side — e.g. because it needs a secret or centrally-managed config the client shouldn't hold — which argues for a small, scoped, defensively-validated endpoint rather than moving the *entire* capability off the server.
- Any time provenance/deduplication keys off "the submitted content" and a refactor changes what's submitted (e.g. markdown instead of a binary): check whether the identity that should be deduplicated is the *original* artifact, not the transformed one, before quietly changing hash semantics.
- Whenever a refactor moves or splits test files: treat "the suite still passes" as necessary but not sufficient — explicitly verify each acceptance criterion still has an assertion that actually exercises the real (non-mocked) code path it was written to protect.

## Examples

**Dockerfile before/after** (conceptual shape of the change):

```dockerfile
# Before
RUN pip install -e ".[ingest]"
RUN apt-get install -y libxcb1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

# After
RUN pip install -e .
# X11/OpenGL packages dropped entirely — they existed only to support
# Docling's image-rendering pipeline, which no longer runs in this container.
```

**pyproject.toml extra renamed to signal ownership**: `ingest` → `prepare`. Docling now lives under an extra that is installed on the CLI machine only, never on the server, and the new name reflects what it actually does (prepare binaries into markdown) rather than the generic "ingest" label that implied it was part of the backend's ingestion responsibility.

**The prepare/finalize split in use** (approximate CLI-side flow):

```python
# rag ingest, suffix-aware:
if suffix in {".md", ".txt"}:
    client.submit_text(content=read(path))
elif suffix in {".pdf", ".docx", ".pptx"}:
    prepared = prepare_binary(path)          # local Docling, CLI machine only
    final_md = finalize_markdown(
        prepared,
        describe_fn=lambda img: client.describe_image(img),  # backend call
    )
    client.submit_text(
        content=final_md,
        original_md5=prepared.original_md5,
        file_name=prepared.file_name,
        file_type=prepared.file_type,
    )
else:
    raise ParseError(f"unsupported suffix: {suffix}")
# If prepare_binary or finalize_markdown raises, nothing is submitted —
# no partially-broken job is ever queued.
```

**Backend rejecting direct binary upload** (`/api/ingest` multipart):

```python
if suffix in {".pdf", ".docx", ".pptx"}:
    raise HTTPException(
        status_code=415,
        detail="Binary documents must be prepared on the CLI first "
               "(see `rag prepare`); direct upload of PDF/DOCX/PPTX "
               "is no longer supported.",
    )
```

**Subprocess import-guard test pattern** (generalizable beyond this repo):

```python
def test_backend_never_imports_docling():
    result = subprocess.run(
        [sys.executable, "-c",
         "import rag.api.main, rag.worker, rag.ingestion; "
         "import sys; "
         "assert 'docling' not in sys.modules, "
         "'docling leaked into backend import graph'"],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
```

**`rag prepare` as a standalone, explicit command**:

```
rag prepare document.pdf --out document.md
```

Converts a binary to markdown without queuing an ingestion job — useful for inspection or pre-generating markdown. If the document contains images, this command requires API config (since captioning is backend-owned) and fails clearly rather than silently skipping captions when that config is absent.

## Related

- `docs/solutions/performance-issues/rag-retrieval-vector-prefilter-and-query-fanout.md` — the earlier documentation that first flagged the Docling/Torch backend bloat as an operational problem (moderate overlap: same problem statement, but that doc's own implemented fix was a binary-quantized HNSW prefilter for vector retrieval and its Prevention section only recommended future Dockerfile layer-reordering — a shallower remedy than the CLI/backend split this doc describes, which removes Docling from the backend runtime entirely). That doc's prevention guidance on this specific point is now superseded by this one.
- `docs/plans/2026-07-03-001-refactor-prepared-binary-ingestion-plan.md` — the 8-unit implementation plan (U1–U8) executed for this fix, including the three decisions left deliberately open before implementation: where Docling code should physically live, how duplicate-detection hashing should work across the prepare/submit split, and what should happen when image description fails.
- `src/rag/parser.py`, `src/rag/prepare.py` — the parse/prepare module split.
- `src/rag/api/auth.py` — `requires_scope()` / `default_principal_dependency` shared-singleton fix, relevant to any future route that needs to be safely bypassable under `app.dependency_overrides` in tests.
- `tests/test_cli_imports.py` — the subprocess Docling-import guard.
- `tests/test_prepare.py`, `tests/test_ingestion.py` — real-Docling pptx test (site of the AE2 regression and its fix) and the rewritten `test_ingest_pdf`/`test_ingest_docx` end-to-end tests.
