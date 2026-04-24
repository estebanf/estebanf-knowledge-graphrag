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
    if (!onAddToBucket) return;
    onAddToBucket(result.source_id, result.source_metadata.title ?? result.source_id);
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
