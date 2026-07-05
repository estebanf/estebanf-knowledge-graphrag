import { useState } from "react";
import type { RetrieveResponse } from "../lib/api";
import { retrieve } from "../lib/api";
import InsightCard from "../components/InsightCard";
import ResultCard from "../components/ResultCard";

type RetrieveViewProps = {
  onView: (sourceId: string) => void;
  onAddToBucket: (sourceId: string, title: string) => void;
  onCopyChunk: (chunk: string) => Promise<void>;
};

export default function RetrieveView({ onView, onAddToBucket, onCopyChunk }: RetrieveViewProps) {
  const [query, setQuery] = useState("");
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
      const response = await retrieve(query.trim());
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
                    <InsightCard result={{ ...seed, topics: seed.topics || [], sources: seed.sources || [] }} onCopy={onCopyChunk} onView={onView} onAddToBucket={onAddToBucket} />
                    {seed.related.map((group) => (
                      <div className="related-group" key={`${seed.insight_id}-${group.type}-${group.sub_query || "first"}`}>
                        <div className="related-group__label">
                          {group.type === "first_hop" ? "Related" : `Related · ${group.sub_query}`}
                        </div>
                        <div className="related-group__stack">
                          {group.insights.map((insight) => (
                            <InsightCard key={insight.insight_id} result={{ ...insight, topics: insight.topics || [], sources: insight.sources || [] }} onCopy={onCopyChunk} onView={onView} onAddToBucket={onAddToBucket} />
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
              <ResultCard result={result} onCopyChunk={onCopyChunk} onView={onView} onAddToBucket={onAddToBucket} />
              {result.related.map((related) => (
                <div className="related-group" key={`${result.chunk_id}-${related.entity}`}>
                  <div className="related-group__label">{related.entity}</div>
                  <div className="related-group__stack">
                    {related.chunks.map((chunk) => (
                      <ResultCard compact key={chunk.chunk_id} result={chunk} onCopyChunk={onCopyChunk} onView={onView} onAddToBucket={onAddToBucket} />
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
