import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

import type { SourceDetail, SourceInsight } from "../lib/api";
import { getSourceInsights } from "../lib/api";

type SourcePanelProps = {
  source: SourceDetail | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
};

type Tab = "content" | "insights";

function chunkLabel(insight: SourceInsight): string {
  if (insight.chunk_index === null || insight.chunk_index === undefined) {
    return "chunk";
  }
  return `chunk ${insight.chunk_index}`;
}

export default function SourcePanel({ source, loading, error, onClose }: SourcePanelProps) {
  const [tab, setTab] = useState<Tab>("content");
  const [insights, setInsights] = useState<SourceInsight[]>([]);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState<string | null>(null);
  const [copiedInsightId, setCopiedInsightId] = useState<string | null>(null);

  useEffect(() => {
    setTab("content");
    setInsights([]);
    setInsightsError(null);
  }, [source?.source_id]);

  useEffect(() => {
    if (tab !== "insights" || !source || insightsLoading || insights.length > 0) return;
    setInsightsLoading(true);
    setInsightsError(null);
    getSourceInsights(source.source_id)
      .then((resp) => setInsights(resp.insights ?? []))
      .catch((e) => setInsightsError(e instanceof Error ? e.message : "Failed to load insights"))
      .finally(() => setInsightsLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, source?.source_id]);

  async function copyInsightText(insight: SourceInsight) {
    await navigator.clipboard.writeText(insight.insight);
    setCopiedInsightId(insight.insight_id);
    window.setTimeout(() => setCopiedInsightId(null), 1200);
  }

  return (
    <aside aria-label="Source Preview" className={`source-panel${source || loading || error ? " source-panel--open" : ""}`} role="complementary">
      <div className="source-panel__frame">
        <div className="source-panel__header">
          <div>
            <p className="eyebrow">Source preview</p>
            <h2>{source?.name || source?.file_name || "Markdown source"}</h2>
          </div>
          <button className="close-button" type="button" onClick={onClose}>
            Close
          </button>
        </div>
        {source && !loading && !error ? (
          <div className="source-detail-tabs" role="tablist" aria-label="Source detail tabs">
            <button
              aria-selected={tab === "content"}
              className={`source-detail-tab${tab === "content" ? " source-detail-tab--active" : ""}`}
              role="tab"
              type="button"
              onClick={() => setTab("content")}
            >
              Content
            </button>
            <button
              aria-selected={tab === "insights"}
              className={`source-detail-tab${tab === "insights" ? " source-detail-tab--active" : ""}`}
              role="tab"
              type="button"
              onClick={() => setTab("insights")}
            >
              Insights
            </button>
          </div>
        ) : null}
        {loading && <p className="panel-state">Loading source...</p>}
        {error && <p className="panel-state panel-state--error">{error}</p>}
        {source && !loading && !error && tab === "content" ? (
          <div className="source-panel__body">
            <div className="source-panel__meta">
              <span>{source.metadata.source || "Local corpus"}</span>
              <span>{source.file_name}</span>
            </div>
            <ReactMarkdown>{source.markdown_content}</ReactMarkdown>
          </div>
        ) : null}
        {source && !loading && !error && tab === "insights" ? (
          <div className="source-panel__body">
            {insightsLoading ? <p className="panel-state">Loading insights...</p> : null}
            {insightsError ? <p className="panel-state panel-state--error">{insightsError}</p> : null}
            {!insightsLoading && !insightsError && insights.length === 0 ? (
              <p className="panel-state">No insights extracted for this source.</p>
            ) : null}
            <div className="source-insights-list">
              {insights.map((insight) => (
                <article className="source-insight-card" key={`${insight.chunk_id}-${insight.insight_id}`}>
                  <div className="source-insight-card__topline">
                    <span>{chunkLabel(insight)}</span>
                    <span>{insight.chunk_id.slice(0, 8)}</span>
                    <div className="feedback-anchor">
                      <button
                        aria-label="Copy insight text"
                        className="icon-button source-copy-button"
                        type="button"
                        onClick={() => copyInsightText(insight)}
                      >
                        ⧉
                      </button>
                      {copiedInsightId === insight.insight_id ? (
                        <span className="copy-popper" role="status">
                          Insight copied
                        </span>
                      ) : null}
                    </div>
                  </div>
                  <p>{insight.insight}</p>
                  <div className="source-insight-topics" aria-label="Connection topics">
                    {insight.topics.length > 0 ? (
                      insight.topics.map((topic) => (
                        <span className="source-insight-topic" key={`${insight.insight_id}-${topic}`}>
                          {topic}
                        </span>
                      ))
                    ) : (
                      <span className="source-insight-topic">extracted</span>
                    )}
                  </div>
                  {insight.chunk_preview ? <blockquote>{insight.chunk_preview}</blockquote> : null}
                </article>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
