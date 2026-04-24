# Source Bucket Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "source bucket" that lets users collect source IDs from result cards and copy them as a newline-separated list for pasting into the Community tab.

**Architecture:** `BucketEntry[]` state lives in `App.tsx` alongside other app state. `ResultCard` receives an optional `onAddToBucket` callback. A new `BucketPopover` component anchored in the topbar shows collected entries and provides clear/copy actions. All elements reuse existing CSS design tokens and patterns (`.icon-button`, `.copy-popper`, `.ghost-button`, etc.).

**Tech Stack:** React 19, TypeScript, Vitest + React Testing Library (integration tests via `App.test.tsx`), custom CSS variables.

---

### Task 1: Add BucketEntry type and bucket state to App.tsx

**Files:**
- Modify: `frontend/src/App.tsx`

**Step 1: Add the type and state**

After the existing `const [resultsCopied, setResultsCopied] = useState(false);` line (~line 26), insert:

```ts
type BucketEntry = { sourceId: string; title: string };
const [bucket, setBucket] = useState<BucketEntry[]>([]);

function handleAddToBucket(sourceId: string, title: string) {
  setBucket((prev) => {
    if (prev.some((e) => e.sourceId === sourceId)) return prev;
    return [...prev, { sourceId, title }];
  });
}

async function handleCopyBucket() {
  const text = bucket.map((e) => e.sourceId).join("\n");
  await navigator.clipboard.writeText(text);
}

function handleClearBucket() {
  setBucket([]);
}
```

**Step 2: Run the existing tests to confirm no regressions**

```bash
cd frontend && npm test
```
Expected: All existing tests pass.

**Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: add source bucket state and handlers to App"
```

---

### Task 2: Add "+ Add" button and "Source added" toast to ResultCard

**Files:**
- Modify: `frontend/src/components/ResultCard.tsx`
- Test: `frontend/src/App.test.tsx`

**Step 1: Write the failing test**

Add inside the `describe("App")` block in `App.test.tsx`:

```ts
test("add to bucket button appears on result cards", async () => {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    json: async () => searchResponse,
  }));
  vi.stubGlobal("fetch", fetchMock);

  render(<App />);
  await userEvent.type(screen.getByLabelText(/semantic query/i), "economics{enter}");
  await screen.findByRole("heading", { name: /Economics of GenAI/i });

  expect(screen.getByRole("button", { name: /add to bucket/i })).toBeInTheDocument();
});
```

**Step 2: Run test to verify it fails**

```bash
cd frontend && npm test
```
Expected: FAIL — "Unable to find an accessible element with the role 'button' and name /add to bucket/i"

**Step 3: Update ResultCard props and add button**

Replace the entire content of `frontend/src/components/ResultCard.tsx` with:

```tsx
import { useState } from "react";

import type { SearchResult } from "../lib/api";

type ResultCardProps = {
  result: SearchResult;
  compact?: boolean;
  onView: (sourceId: string) => void;
  onCopyChunk: (chunk: string) => Promise<void>;
  onAddToBucket?: (sourceId: string, title: string) => void;
};

function badgeLabel(result: SearchResult): string {
  return result.source_metadata.kind || "source";
}

function sourceLabel(result: SearchResult): string {
  return result.source_metadata.source || result.source_metadata.author || "Unknown source";
}

export default function ResultCard({ result, compact = false, onView, onCopyChunk, onAddToBucket }: ResultCardProps) {
  const title = result.source_metadata.title;
  const [copied, setCopied] = useState(false);
  const [added, setAdded] = useState(false);

  async function handleCopyChunk() {
    await onCopyChunk(result.chunk);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  function handleAddToBucket() {
    onAddToBucket?.(result.source_id, result.source_metadata.title ?? result.source_id);
    setAdded(true);
    window.setTimeout(() => setAdded(false), 1200);
  }

  return (
    <article className={`result-card${compact ? " result-card--compact" : ""}`}>
      <div className="result-card__header">
        <div>
          <div className="result-card__meta">
            <span className="badge">{badgeLabel(result)}</span>
            <span>{sourceLabel(result)}</span>
          </div>
          {title ? <h4 className="result-card__title">{title}</h4> : null}
          <div className="result-card__chunk-row">
            <p className="result-card__text">{result.chunk}</p>
            <div className="feedback-anchor">
              <button
                aria-label="Copy chunk"
                className="icon-button"
                type="button"
                onClick={handleCopyChunk}
              >
                ⧉
              </button>
              {copied ? (
                <span className="copy-popper" role="status">
                  Chunk copied
                </span>
              ) : null}
            </div>
          </div>
        </div>
        <span className="score-chip">Score: {result.score.toFixed(3)}</span>
      </div>
      <div className="result-card__actions">
        <div className="feedback-anchor">
          <button
            aria-label="Add to bucket"
            className="ghost-button"
            type="button"
            onClick={handleAddToBucket}
          >
            + Add
          </button>
          {added ? (
            <span className="copy-popper" role="status">
              Source added
            </span>
          ) : null}
        </div>
        <button className="ghost-button" type="button" onClick={() => onView(result.source_id)}>
          View
        </button>
        <a className="ghost-button ghost-button--muted" href={`/api/sources/${result.source_id}/download`}>
          Download
        </a>
      </div>
    </article>
  );
}
```

**Step 4: Run tests**

```bash
cd frontend && npm test
```
Expected: "add to bucket button appears" passes. All other tests pass.

**Step 5: Commit**

```bash
git add frontend/src/components/ResultCard.tsx frontend/src/App.test.tsx
git commit -m "feat: add + Add button and Source added toast to ResultCard"
```

---

### Task 3: Wire onAddToBucket into all ResultCard usages in App.tsx

**Files:**
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

**Step 1: Write the failing dedup test**

Add to `App.test.tsx`:

```ts
test("adding same source twice keeps only one bucket entry", async () => {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    json: async () => searchResponse,
  }));
  vi.stubGlobal("fetch", fetchMock);

  render(<App />);
  await userEvent.type(screen.getByLabelText(/semantic query/i), "economics{enter}");
  await screen.findByRole("heading", { name: /Economics of GenAI/i });

  const addBtn = screen.getByRole("button", { name: /add to bucket/i });
  await userEvent.click(addBtn);
  await userEvent.click(addBtn);

  await userEvent.click(screen.getByRole("button", { name: /source bucket/i }));
  const entries = screen.getAllByTestId("bucket-entry");
  expect(entries).toHaveLength(1);
});
```

**Step 2: Run test to verify it fails**

```bash
cd frontend && npm test
```
Expected: FAIL — "source bucket" button not found yet.

**Step 3: Add onAddToBucket to all three ResultCard usages**

In `App.tsx`, find the three `<ResultCard` usages and add `onAddToBucket={handleAddToBucket}` to each:

1. Search results map (~line 514):
```tsx
<ResultCard key={result.chunk_id} result={result} onCopyChunk={copyChunk} onView={handleView} onAddToBucket={handleAddToBucket} />
```

2. Retrieve root cards (~line 520):
```tsx
<ResultCard result={result} onCopyChunk={copyChunk} onView={handleView} onAddToBucket={handleAddToBucket} />
```

3. Retrieve related compact cards (~line 526):
```tsx
<ResultCard compact key={chunk.chunk_id} result={chunk} onCopyChunk={copyChunk} onView={handleView} onAddToBucket={handleAddToBucket} />
```

**Step 4: Run tests**

```bash
cd frontend && npm test
```
Expected: Dedup test still fails (no bucket popover yet). All other tests pass.

**Step 5: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: wire onAddToBucket into all ResultCard usages"
```

---

### Task 4: Create BucketPopover component

**Files:**
- Create: `frontend/src/components/BucketPopover.tsx`

**Step 1: Create the component**

```tsx
import { useEffect, useRef, useState } from "react";

type BucketEntry = { sourceId: string; title: string };

type BucketPopoverProps = {
  entries: BucketEntry[];
  onClear: () => void;
  onCopy: () => Promise<void>;
};

export default function BucketPopover({ entries, onClear, onCopy }: BucketPopoverProps) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleOutsideClick(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [open]);

  async function handleCopy() {
    await onCopy();
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="bucket-anchor" ref={containerRef}>
      <button
        aria-label="Source bucket"
        className="icon-button"
        type="button"
        onClick={() => setOpen((prev) => !prev)}
      >
        ⊟
        {entries.length > 0 ? (
          <span className="bucket-count">{entries.length}</span>
        ) : null}
      </button>
      {open ? (
        <div aria-label="Collected sources" className="bucket-popover" role="dialog">
          <p className="bucket-popover__label">
            {entries.length === 0
              ? "No sources collected"
              : `${entries.length} source${entries.length === 1 ? "" : "s"}`}
          </p>
          {entries.length > 0 ? (
            <ul className="bucket-popover__list">
              {entries.map((e) => (
                <li className="bucket-popover__item" data-testid="bucket-entry" key={e.sourceId}>
                  <span className="bucket-popover__title">{e.title}</span>
                  <span className="bucket-popover__id">{e.sourceId}</span>
                </li>
              ))}
            </ul>
          ) : null}
          <div className="bucket-popover__actions">
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                onClear();
                setOpen(false);
              }}
            >
              Clear
            </button>
            <div className="feedback-anchor">
              <button
                className="primary-button"
                disabled={entries.length === 0}
                type="button"
                onClick={handleCopy}
              >
                Copy
              </button>
              {copied ? (
                <span className="copy-popper" role="status">
                  IDs copied
                </span>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
```

**Step 2: Verify the file created without errors**

```bash
cd frontend && npx tsc --noEmit
```
Expected: No errors.

**Step 3: Commit**

```bash
git add frontend/src/components/BucketPopover.tsx
git commit -m "feat: create BucketPopover component"
```

---

### Task 5: Wire BucketPopover into App.tsx topbar and add remaining tests

**Files:**
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

**Step 1: Import BucketPopover in App.tsx**

Add to the imports at the top of `App.tsx`:

```ts
import BucketPopover from "./components/BucketPopover";
```

**Step 2: Add BucketPopover to the topbar**

Replace the existing `<header className="topbar">` block with:

```tsx
<header className="topbar">
  <div>
    <p className="eyebrow">Knowledge graph retrieval</p>
    <div className="topbar__title-row">
      <h1>estebanf&apos;s RAG</h1>
    </div>
  </div>
  <BucketPopover
    entries={bucket}
    onClear={handleClearBucket}
    onCopy={handleCopyBucket}
  />
</header>
```

**Step 3: Run tests**

```bash
cd frontend && npm test
```
Expected: Dedup test now passes. All other tests pass.

**Step 4: Add copy and clear tests**

Add to `App.test.tsx`:

```ts
test("bucket popover lists collected sources and copies IDs one per line", async () => {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    json: async () => searchResponse,
  }));
  const writeText = installClipboard();
  vi.stubGlobal("fetch", fetchMock);

  render(<App />);
  await userEvent.type(screen.getByLabelText(/semantic query/i), "economics{enter}");
  await screen.findByRole("heading", { name: /Economics of GenAI/i });

  await userEvent.click(screen.getByRole("button", { name: /add to bucket/i }));
  await userEvent.click(screen.getByRole("button", { name: /source bucket/i }));

  expect(screen.getByRole("dialog", { name: /collected sources/i })).toBeInTheDocument();
  expect(screen.getByTestId("bucket-entry")).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: /^copy$/i }));
  expect(writeText).toHaveBeenCalledWith("source-1");
  expect(screen.getByText(/ids copied/i)).toBeInTheDocument();
});

test("bucket clear empties the collected sources", async () => {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    json: async () => searchResponse,
  }));
  vi.stubGlobal("fetch", fetchMock);

  render(<App />);
  await userEvent.type(screen.getByLabelText(/semantic query/i), "economics{enter}");
  await screen.findByRole("heading", { name: /Economics of GenAI/i });

  await userEvent.click(screen.getByRole("button", { name: /add to bucket/i }));
  await userEvent.click(screen.getByRole("button", { name: /source bucket/i }));
  await userEvent.click(screen.getByRole("button", { name: /clear/i }));

  // Popover closes after clear; reopen to verify empty state
  await userEvent.click(screen.getByRole("button", { name: /source bucket/i }));
  expect(screen.getByText(/no sources collected/i)).toBeInTheDocument();
  expect(screen.queryByTestId("bucket-entry")).not.toBeInTheDocument();
});
```

**Step 5: Run all tests**

```bash
cd frontend && npm test
```
Expected: All tests pass.

**Step 6: Commit**

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat: wire BucketPopover into topbar and add integration tests"
```

---

### Task 6: Add CSS styles for the bucket button, count badge, and popover

**Files:**
- Modify: `frontend/src/styles/app.css`

**Step 1: Append bucket styles at the end of app.css**

```css
/* Source bucket */

.bucket-anchor {
  position: relative;
  display: inline-flex;
  align-self: center;
}

.bucket-count {
  position: absolute;
  top: -0.3rem;
  right: -0.3rem;
  min-width: 1.1rem;
  height: 1.1rem;
  padding: 0 0.25rem;
  border-radius: 999px;
  background: var(--warm);
  color: #fef9f3;
  font-size: 0.65rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
}

.bucket-popover {
  position: absolute;
  top: calc(100% + 0.6rem);
  right: 0;
  width: 22rem;
  padding: 1rem;
  background: var(--panel-strong);
  border: 1px solid var(--border-strong);
  border-radius: 1.3rem;
  box-shadow: var(--shadow);
  z-index: 100;
}

.bucket-popover__label {
  margin: 0 0 0.75rem;
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.bucket-popover__list {
  list-style: none;
  margin: 0 0 0.85rem;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  max-height: 14rem;
  overflow-y: auto;
}

.bucket-popover__item {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  padding: 0.55rem 0.7rem;
  background: var(--bg-soft);
  border-radius: 0.7rem;
}

.bucket-popover__title {
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text);
}

.bucket-popover__id {
  font-size: 0.72rem;
  color: var(--muted);
  font-family: monospace;
  word-break: break-all;
}

.bucket-popover__actions {
  display: flex;
  gap: 0.65rem;
  justify-content: flex-end;
}

.bucket-popover__actions .secondary-button,
.bucket-popover__actions .primary-button {
  padding: 0.65rem 1rem;
  font-size: 0.88rem;
}
```

**Step 2: Run tests**

```bash
cd frontend && npm test
```
Expected: All tests pass.

**Step 3: Commit**

```bash
git add frontend/src/styles/app.css
git commit -m "feat: add CSS styles for source bucket popover and count badge"
```

---

### Verification

After all tasks complete, manually verify in the browser (`npm run dev`):
- "+ Add" button appears on every result card in Search, Retrieve (root and compact)
- Clicking it shows "Source added" toast for 1.2 s
- Adding the same source twice does not duplicate it
- Bucket icon appears in top-right of the topbar
- Count badge increments as sources are added
- Clicking the icon opens the popover listing titles + IDs
- "Clear" closes the popover and empties the bucket
- "Copy" writes IDs one per line to clipboard and shows "IDs copied" for 1.2 s
- Clicking outside the popover closes it
