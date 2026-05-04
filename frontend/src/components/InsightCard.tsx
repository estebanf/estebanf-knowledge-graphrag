import { useState } from "react";

import type { InsightResult } from "../lib/api";

type InsightCardProps = {
  result: InsightResult;
  onView: (sourceId: string) => void;
  onCopy: (text: string) => Promise<void>;
  onAddToBucket?: (sourceId: string, title: string) => void;
};

function primarySource(result: InsightResult) {
  return result.sources[0];
}

function sourceLabel(result: InsightResult): string {
  const src = primarySource(result);
  if (!src) return "Unknown source";
  return src.source_metadata.source || src.source_metadata.author || "Unknown source";
}

export default function InsightCard({ result, onView, onCopy, onAddToBucket }: InsightCardProps) {
  const src = primarySource(result);
  const title = src?.source_metadata.title;
  const kind = src?.source_metadata.kind;
  const [copied, setCopied] = useState(false);
  const [added, setAdded] = useState(false);

  async function handleCopy() {
    await onCopy(result.insight);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  function handleAddToBucket() {
    if (!onAddToBucket || !src) return;
    onAddToBucket(src.source_id, src.source_metadata.title ?? src.source_id);
    setAdded(true);
    window.setTimeout(() => setAdded(false), 1200);
  }

  return (
    <article className="result-card">
      <div className="result-card__header">
        <div>
          <div className="result-card__meta">
            <span className="badge">
              {kind ? `insight · ${kind}` : "insight"}
            </span>
            <span>{sourceLabel(result)}</span>
            {result.sources.length > 1 ? (
              <span className="result-card__source-count">
                +{result.sources.length - 1} more source{result.sources.length > 2 ? "s" : ""}
              </span>
            ) : null}
          </div>
          {title ? <h4 className="result-card__title">{title}</h4> : null}
          {result.topics.length > 0 ? (
            <div className="result-card__topics">
              {result.topics.map((t) => (
                <span key={t} className="topic-tag">
                  {t}
                </span>
              ))}
            </div>
          ) : null}
          <div className="result-card__chunk-row">
            <p className="result-card__text">{result.insight}</p>
            <div className="feedback-anchor">
              <button
                aria-label="Copy insight"
                className="icon-button"
                type="button"
                onClick={handleCopy}
              >
                ⧉
              </button>
              {copied ? (
                <span className="copy-popper" role="status">
                  Insight copied
                </span>
              ) : null}
            </div>
          </div>
        </div>
        <span className="score-chip">Score: {result.score.toFixed(3)}</span>
      </div>
      {src ? (
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
          <button className="ghost-button" type="button" onClick={() => onView(src.source_id)}>
            View
          </button>
          <a className="ghost-button ghost-button--muted" href={`/api/sources/${src.source_id}/download`}>
            Download
          </a>
        </div>
      ) : null}
    </article>
  );
}
