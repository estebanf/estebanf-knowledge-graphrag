import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

import type { MetadataFilter, SourceDetail, SourceInsight, SourceSummary } from "../lib/api";

type SourcesExplorerProps = {
  sources: SourceSummary[];
  selectedSourceId: string | null;
  selectedFilters: MetadataFilter[];
  page: number;
  pageCount: number;
  source: SourceDetail | null;
  insights: SourceInsight[];
  loadingSources: boolean;
  loadingSource: boolean;
  loadingInsights: boolean;
  sourcesError: string | null;
  sourceError: string | null;
  insightsError: string | null;
  onSelectSource: (sourceId: string) => void;
  onToggleMetadataFilter: (filter: MetadataFilter) => void;
  onRemoveMetadataFilter: (filter: MetadataFilter) => void;
  onNextPage: () => void;
  onPreviousPage: () => void;
};

type DetailTab = "insights" | "content";

function sourceTitle(source: SourceSummary | SourceDetail | null): string {
  if (!source) return "No source selected";
  return source.name || source.file_name || source.source_id;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function chunkLabel(insight: SourceInsight): string {
  if (insight.chunk_index === null || insight.chunk_index === undefined) {
    return "chunk";
  }
  return `chunk ${insight.chunk_index}`;
}

function metadataEntries(metadata: SourceSummary["metadata"]): MetadataFilter[] {
  return Object.entries(metadata)
    .filter(([, value]) => value !== null && value !== undefined && String(value).trim() !== "")
    .slice(0, 6)
    .map(([key, value]) => ({ key, value: String(value) }));
}

function filterLabel(filter: MetadataFilter): string {
  return `${filter.key} ${filter.value}`;
}

export default function SourcesExplorer({
  sources,
  selectedSourceId,
  selectedFilters,
  page,
  pageCount,
  source,
  insights,
  loadingSources,
  loadingSource,
  loadingInsights,
  sourcesError,
  sourceError,
  insightsError,
  onSelectSource,
  onToggleMetadataFilter,
  onRemoveMetadataFilter,
  onNextPage,
  onPreviousPage,
}: SourcesExplorerProps) {
  const [activeTab, setActiveTab] = useState<DetailTab>("content");
  const [copiedSourceInsights, setCopiedSourceInsights] = useState(false);
  const [copiedInsightId, setCopiedInsightId] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedSourceId || loadingInsights) {
      return;
    }
    setActiveTab(insights.length > 0 ? "insights" : "content");
  }, [selectedSourceId, loadingInsights, insights.length]);

  async function copySourceInsights() {
    await navigator.clipboard.writeText(JSON.stringify(insights, null, 2));
    setCopiedSourceInsights(true);
    window.setTimeout(() => setCopiedSourceInsights(false), 1200);
  }

  async function copyInsightText(insight: SourceInsight) {
    await navigator.clipboard.writeText(insight.insight);
    setCopiedInsightId(insight.insight_id);
    window.setTimeout(() => setCopiedInsightId(null), 1200);
  }

  return (
    <section className="sources-explorer" aria-label="Sources explorer">
      <aside className="sources-pane sources-pane--list" aria-label="Recent sources">
        <div className="sources-pane__header">
          <div>
            <p className="eyebrow">Sources</p>
            <h2>Latest 20</h2>
          </div>
          <span>{sources.length}</span>
        </div>

        {selectedFilters.length > 0 ? (
          <div className="source-filter-bar" aria-label="Selected filters">
            {selectedFilters.map((filter) => (
              <button
                aria-label={`Remove filter ${filterLabel(filter)}`}
                className="source-filter-chip"
                key={`${filter.key}:${filter.value}`}
                type="button"
                onClick={() => onRemoveMetadataFilter(filter)}
              >
                <span>{filter.key}</span>
                <strong>{filter.value}</strong>
              </button>
            ))}
          </div>
        ) : null}

        {loadingSources ? <p className="panel-state">Loading sources...</p> : null}
        {sourcesError ? <p className="panel-state panel-state--error">{sourcesError}</p> : null}
        {!loadingSources && !sourcesError && sources.length === 0 ? (
          <p className="panel-state">No sources found.</p>
        ) : null}

        <div className="source-list">
          {sources.map((item) => (
            <article
              className={`source-list__item${item.source_id === selectedSourceId ? " source-list__item--active" : ""}`}
              key={item.source_id}
            >
              <button className="source-list__select" type="button" onClick={() => onSelectSource(item.source_id)}>
                <span className="source-list__title">{sourceTitle(item)}</span>
                <span className="source-list__meta">
                  <span>
                    {formatDate(item.created_at)}
                    {item.insight_count > 0 ? ` · ${item.insight_count} insight${item.insight_count === 1 ? "" : "s"}` : ""}
                  </span>
                  <span className="source-list__id">{item.source_id}</span>
                </span>
              </button>
              {metadataEntries(item.metadata).length > 0 ? (
                <div className="source-list__attributes" aria-label="Source metadata">
                  {metadataEntries(item.metadata).map((filter) => (
                    <button
                      aria-label={`Filter by ${filterLabel(filter)}`}
                      className="source-list__attribute"
                      key={`${item.source_id}-${filter.key}`}
                      type="button"
                      onClick={() => onToggleMetadataFilter(filter)}
                    >
                      <span>{filter.key}</span>
                      <strong>{filter.value}</strong>
                    </button>
                  ))}
                </div>
              ) : null}
            </article>
          ))}
        </div>

        <div className="source-paginator" aria-label="Sources pagination">
          <button
            className="source-paginator__button"
            disabled={page <= 1 || loadingSources}
            type="button"
            onClick={onPreviousPage}
          >
            Previous
          </button>
          <span>Page {page} of {pageCount}</span>
          <button
            aria-label="Next page"
            className="source-paginator__button"
            disabled={page >= pageCount || loadingSources}
            type="button"
            onClick={onNextPage}
          >
            Next
          </button>
        </div>
      </aside>

      <section className="sources-pane sources-pane--detail" aria-label="Source detail">
        <div className="sources-pane__header sources-pane__header--detail">
          <div>
            <p className="eyebrow">Source</p>
            <div className="source-detail-title-row">
              <h2>{sourceTitle(source)}</h2>
              <div className="feedback-anchor">
                <button
                  aria-label="Copy source insights"
                  className="icon-button source-copy-button"
                  disabled={!selectedSourceId || loadingInsights}
                  type="button"
                  onClick={copySourceInsights}
                >
                  ⧉
                </button>
                {copiedSourceInsights ? (
                  <span className="copy-popper" role="status">
                    Insights copied
                  </span>
                ) : null}
              </div>
            </div>
          </div>
          <div className="source-detail-tabs" role="tablist" aria-label="Source detail tabs">
            <button
              aria-selected={activeTab === "insights"}
              className={`source-detail-tab${activeTab === "insights" ? " source-detail-tab--active" : ""}`}
              role="tab"
              type="button"
              onClick={() => setActiveTab("insights")}
            >
              Insights
            </button>
            <button
              aria-selected={activeTab === "content"}
              className={`source-detail-tab${activeTab === "content" ? " source-detail-tab--active" : ""}`}
              role="tab"
              type="button"
              onClick={() => setActiveTab("content")}
            >
              Content
            </button>
          </div>
        </div>

        {activeTab === "insights" ? (
          <div className="source-detail-body">
            {loadingInsights ? <p className="panel-state">Loading insights...</p> : null}
            {insightsError ? <p className="panel-state panel-state--error">{insightsError}</p> : null}
            {!loadingInsights && !insightsError && selectedSourceId && insights.length === 0 ? (
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
        ) : (
          <div className="source-detail-body">
            {loadingSource ? <p className="panel-state">Loading markdown...</p> : null}
            {sourceError ? <p className="panel-state panel-state--error">{sourceError}</p> : null}
            {source && !loadingSource && !sourceError ? (
              <div className="source-markdown">
                <div className="source-markdown__meta">
                  <span>{source.metadata.source || "Local corpus"}</span>
                  <span>{source.file_name}</span>
                </div>
                <ReactMarkdown>{source.markdown_content || "_No markdown content._"}</ReactMarkdown>
              </div>
            ) : null}
            {!source && !loadingSource && !sourceError ? (
              <p className="panel-state">Select a source to render its markdown.</p>
            ) : null}
          </div>
        )}
      </section>
    </section>
  );
}
