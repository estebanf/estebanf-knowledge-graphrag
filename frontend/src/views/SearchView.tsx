import { useState } from "react";
import type { SearchResponse } from "../lib/api";
import { search } from "../lib/api";
import InsightCard from "../components/InsightCard";
import ResultCard from "../components/ResultCard";

type SearchViewProps = {
  onView: (sourceId: string) => void;
  onAddToBucket: (sourceId: string, title: string) => void;
  onCopyChunk: (chunk: string) => Promise<void>;
};

export default function SearchView({ onView, onAddToBucket, onCopyChunk }: SearchViewProps) {
  const [query, setQuery] = useState("");
  const [minScore, setMinScore] = useState("0.7");
  const [limit, setLimit] = useState("10");
  const [results, setResults] = useState<SearchResponse["results"]>({ chunks: [], insights: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultsCopied, setResultsCopied] = useState(false);

  async function handleSubmit() {
    if (!query.trim()) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await search(query.trim(), Number.parseInt(limit, 10), Number.parseFloat(minScore));
      setResults(response.results);
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
    setResults({ chunks: [], insights: [] });
    setError(null);
  }

  async function copyResults() {
    await navigator.clipboard.writeText(JSON.stringify(results, null, 2));
    setResultsCopied(true);
    window.setTimeout(() => setResultsCopied(false), 1200);
  }

  const count = results.chunks.length + results.insights.length;

  return (
    <div className="content">
      <form className="query-panel" onSubmit={handleFormSubmit} aria-label="Search form">
        <label className="query-panel__label" htmlFor="semantic-query">
          Semantic Query
        </label>
        <div className="search-controls">
          <label className="search-controls__field" htmlFor="minimum-score">
            <span>Minimum Score</span>
            <input
              id="minimum-score"
              inputMode="decimal"
              max="1"
              min="0"
              step="0.01"
              type="number"
              value={minScore}
              onChange={(e) => setMinScore(e.target.value)}
            />
          </label>
          <label className="search-controls__field" htmlFor="result-count">
            <span>Result Count</span>
            <input
              id="result-count"
              inputMode="numeric"
              min="1"
              step="1"
              type="number"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
            />
          </label>
        </div>
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
              Search
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
          <span>{isLoading ? "Loading..." : `${count} result${count === 1 ? "" : "s"}`}</span>
        </div>

        {error ? <p className="panel-state panel-state--error">{error}</p> : null}

        <div className="results-sections">
          {results.insights.length > 0 ? (
            <section className="results-section">
              <h3 className="results-section__heading">Insights</h3>
              <div className="results-stack">
                {results.insights.map((result) => (
                  <InsightCard
                    key={result.insight_id}
                    result={result}
                    onCopy={onCopyChunk}
                    onView={onView}
                    onAddToBucket={onAddToBucket}
                  />
                ))}
              </div>
            </section>
          ) : null}
          {results.chunks.length > 0 ? (
            <section className="results-section">
              <h3 className="results-section__heading">Chunks</h3>
              <div className="results-stack">
                {results.chunks.map((result) => (
                  <ResultCard
                    key={result.chunk_id}
                    result={result}
                    onCopyChunk={onCopyChunk}
                    onView={onView}
                    onAddToBucket={onAddToBucket}
                  />
                ))}
              </div>
            </section>
          ) : null}
          {results.insights.length === 0 && results.chunks.length === 0 ? (
            <p className="panel-state">No results found.</p>
          ) : null}
        </div>
      </section>
    </div>
  );
}
