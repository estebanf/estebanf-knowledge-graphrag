import { useState } from "react";
import type { RetrieveResponse } from "../lib/api";
import { retrieve } from "../lib/api";
import InsightCard from "../components/InsightCard";
import ResultCard from "../components/ResultCard";
import argHelp from "../lib/argHelp";

type RetrieveViewProps = {
  onView: (sourceId: string) => void;
  onCopyChunk: (chunk: string) => Promise<void>;
};

function parseSourceIds(raw: string): string[] {
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function parseFilters(raw: string): Record<string, string> {
  const filters: Record<string, string> = {};
  for (const pair of raw.split(",")) {
    const [key, ...rest] = pair.split("=");
    const trimmedKey = key?.trim();
    const value = rest.join("=").trim();
    if (trimmedKey && value) filters[trimmedKey] = value;
  }
  return filters;
}

export default function RetrieveView({ onView, onCopyChunk }: RetrieveViewProps) {
  const [query, setQuery] = useState("");
  const [sourceIds, setSourceIds] = useState("");
  const [filters, setFilters] = useState("");
  const [seedCount, setSeedCount] = useState("10");
  const [resultCount, setResultCount] = useState("5");
  const [rrfK, setRrfK] = useState("60");
  const [entityConfidenceThreshold, setEntityConfidenceThreshold] = useState("0.75");
  const [firstHopSimilarityThreshold, setFirstHopSimilarityThreshold] = useState("0.5");
  const [secondHopSimilarityThreshold, setSecondHopSimilarityThreshold] = useState("0.5");
  const [trace, setTrace] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [results, setResults] = useState<RetrieveResponse["retrieval_results"]>([]);
  const [insights, setInsights] = useState<RetrieveResponse["insights"]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultsCopied, setResultsCopied] = useState(false);

  async function handleSubmit() {
    if (!query.trim()) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await retrieve(query.trim(), {
        source_ids: parseSourceIds(sourceIds),
        filters: parseFilters(filters),
        seed_count: Number.parseInt(seedCount, 10),
        result_count: Number.parseInt(resultCount, 10),
        rrf_k: Number.parseInt(rrfK, 10),
        entity_confidence_threshold: Number.parseFloat(entityConfidenceThreshold),
        first_hop_similarity_threshold: Number.parseFloat(firstHopSimilarityThreshold),
        second_hop_similarity_threshold: Number.parseFloat(secondHopSimilarityThreshold),
        trace,
      });
      setResults(response.retrieval_results);
      setInsights(response.insights || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown request error");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleFormSubmit(event?: React.FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    await handleSubmit();
  }

  function handleClear() {
    setQuery("");
    setResults([]);
    setInsights([]);
    setError(null);
  }

  async function copyResults() {
    await navigator.clipboard.writeText(JSON.stringify(results, null, 2));
    setResultsCopied(true);
    window.setTimeout(() => setResultsCopied(false), 1200);
  }

  return (
    <div className="content">
      <form className="query-panel" onSubmit={handleFormSubmit} aria-label="Retrieve form">
        <label className="query-panel__label" htmlFor="semantic-query">
          Semantic Query
        </label>
        <div className="query-panel__controls">
          <input
            id="semantic-query"
            placeholder="Search across documents and research models..."
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="query-panel__actions">
            <button className="secondary-button" type="button" onClick={handleClear}>
              Clear
            </button>
            <button className="primary-button" type="submit">
              Retrieve
            </button>
          </div>
        </div>

        <div className="advanced-disclosure">
          <button
            className="ghost-button"
            type="button"
            onClick={() => setShowAdvanced((prev) => !prev)}
            aria-expanded={showAdvanced}
          >
            {showAdvanced ? "▾ Advanced" : "▸ Advanced"}
          </button>
        </div>

        {showAdvanced ? (
          <div className="search-controls community-options-grid">
            <label className="search-controls__field" htmlFor="retrieve-source-ids">
              <span>Source IDs</span>
              <input
                id="retrieve-source-ids"
                placeholder="src_123, src_456"
                type="text"
                value={sourceIds}
                onChange={(e) => setSourceIds(e.target.value)}
              />
              <span className="arg-help">{argHelp.retrieve_source_ids}</span>
            </label>
            <label className="search-controls__field" htmlFor="retrieve-filters">
              <span>Filters</span>
              <input
                id="retrieve-filters"
                placeholder="author=Jane, kind=article"
                type="text"
                value={filters}
                onChange={(e) => setFilters(e.target.value)}
              />
              <span className="arg-help">{argHelp.retrieve_filters}</span>
            </label>
            <label className="search-controls__field" htmlFor="retrieve-seed-count">
              <span>Seed Count</span>
              <input
                id="retrieve-seed-count"
                min="1"
                step="1"
                type="number"
                value={seedCount}
                onChange={(e) => setSeedCount(e.target.value)}
              />
              <span className="arg-help">{argHelp.retrieve_seed_count}</span>
            </label>
            <label className="search-controls__field" htmlFor="retrieve-result-count">
              <span>Result Count</span>
              <input
                id="retrieve-result-count"
                min="1"
                step="1"
                type="number"
                value={resultCount}
                onChange={(e) => setResultCount(e.target.value)}
              />
              <span className="arg-help">{argHelp.retrieve_result_count}</span>
            </label>
            <label className="search-controls__field" htmlFor="retrieve-rrf-k">
              <span>RRF K</span>
              <input
                id="retrieve-rrf-k"
                min="1"
                step="1"
                type="number"
                value={rrfK}
                onChange={(e) => setRrfK(e.target.value)}
              />
              <span className="arg-help">{argHelp.retrieve_rrf_k}</span>
            </label>
            <label className="search-controls__field" htmlFor="retrieve-entity-confidence">
              <span>Entity Confidence Threshold</span>
              <input
                id="retrieve-entity-confidence"
                max="1"
                min="0"
                step="0.01"
                type="number"
                value={entityConfidenceThreshold}
                onChange={(e) => setEntityConfidenceThreshold(e.target.value)}
              />
              <span className="arg-help">{argHelp.retrieve_entity_confidence_threshold}</span>
            </label>
            <label className="search-controls__field" htmlFor="retrieve-first-hop">
              <span>First-Hop Similarity Threshold</span>
              <input
                id="retrieve-first-hop"
                max="1"
                min="0"
                step="0.01"
                type="number"
                value={firstHopSimilarityThreshold}
                onChange={(e) => setFirstHopSimilarityThreshold(e.target.value)}
              />
              <span className="arg-help">{argHelp.retrieve_first_hop_similarity_threshold}</span>
            </label>
            <label className="search-controls__field" htmlFor="retrieve-second-hop">
              <span>Second-Hop Similarity Threshold</span>
              <input
                id="retrieve-second-hop"
                max="1"
                min="0"
                step="0.01"
                type="number"
                value={secondHopSimilarityThreshold}
                onChange={(e) => setSecondHopSimilarityThreshold(e.target.value)}
              />
              <span className="arg-help">{argHelp.retrieve_second_hop_similarity_threshold}</span>
            </label>
            <div className="search-controls__field">
              <label className="community-summarize-label" htmlFor="retrieve-trace">
                <input
                  id="retrieve-trace"
                  checked={trace}
                  type="checkbox"
                  onChange={(e) => setTrace(e.target.checked)}
                />
                <span>Include Trace</span>
              </label>
              <span className="arg-help">{argHelp.retrieve_trace}</span>
            </div>
          </div>
        ) : null}
      </form>

      <section className="results-panel">
        <div className="results-panel__header">
          <div className="results-panel__title-row">
            <h3>Top Results</h3>
            <div className="feedback-anchor">
              <button aria-label="Copy Results" className="icon-button" type="button" onClick={copyResults}>
                ⧉
              </button>
              {resultsCopied ? (
                <span className="copy-popper" role="status">
                  Results copied
                </span>
              ) : null}
            </div>
          </div>
          <span>{isLoading ? "Loading..." : `${results.length} result${results.length === 1 ? "" : "s"}`}</span>
        </div>

        {error ? <p className="panel-state panel-state--error">{error}</p> : null}

        <div className="results-stack">
          {insights.length > 0 ? (
            <section className="results-section">
              <h3 className="results-section__heading">Insights</h3>
              <div className="results-stack">
                {insights.map((seed) => (
                  <section className="retrieve-group" key={seed.insight_id}>
                    <InsightCard result={{ ...seed, topics: seed.topics || [], sources: seed.sources || [] }} onCopy={onCopyChunk} onView={onView} />
                    {seed.related.map((group) => (
                      <div className="related-group" key={`${seed.insight_id}-${group.type}-${group.sub_query || "first"}`}>
                        <div className="related-group__label">
                          {group.type === "first_hop" ? "Related" : `Related · ${group.sub_query}`}
                        </div>
                        <div className="related-group__stack">
                          {group.insights.map((insight) => (
                            <InsightCard key={insight.insight_id} result={{ ...insight, topics: insight.topics || [], sources: insight.sources || [] }} onCopy={onCopyChunk} onView={onView} />
                          ))}
                        </div>
                      </div>
                    ))}
                  </section>
                ))}
              </div>
            </section>
          ) : null}
          {results.map((result) => (
            <section className="retrieve-group" key={result.chunk_id}>
              <ResultCard result={result} onCopyChunk={onCopyChunk} onView={onView} />
              {result.related.map((related) => (
                <div className="related-group" key={`${result.chunk_id}-${related.entity}`}>
                  <div className="related-group__label">{related.entity}</div>
                  <div className="related-group__stack">
                    {related.chunks.map((chunk) => (
                      <ResultCard compact key={chunk.chunk_id} result={chunk} onCopyChunk={onCopyChunk} onView={onView} />
                    ))}
                  </div>
                </div>
              ))}
            </section>
          ))}
        </div>
      </section>
    </div>
  );
}
