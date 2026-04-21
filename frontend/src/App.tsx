import { useEffect, useState } from "react";

import ResultCard from "./components/ResultCard";
import SourcePanel from "./components/SourcePanel";
import type { AnswerModel, RetrieveResponse, SearchResponse, SourceDetail } from "./lib/api";
import { getAnswerModels, getSource, retrieve, search, streamAnswer } from "./lib/api";

type Mode = "search" | "retrieve" | "answer";

export default function App() {
  const [mode, setMode] = useState<Mode>("search");
  const [query, setQuery] = useState("");
  const [searchMinScore, setSearchMinScore] = useState("0.7");
  const [searchLimit, setSearchLimit] = useState("10");
  const [searchResults, setSearchResults] = useState<SearchResponse["results"]>([]);
  const [retrieveResults, setRetrieveResults] = useState<RetrieveResponse["retrieval_results"]>([]);
  const [answerText, setAnswerText] = useState("");
  const [answerModels, setAnswerModels] = useState<AnswerModel[]>([]);
  const [selectedAnswerModel, setSelectedAnswerModel] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewSource, setPreviewSource] = useState<SourceDetail | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [resultsCopied, setResultsCopied] = useState(false);

  useEffect(() => {
    if (mode !== "answer" || answerModels.length > 0) {
      return;
    }

    let active = true;
    getAnswerModels()
      .then((models) => {
        if (!active) {
          return;
        }
        setAnswerModels(models);
        const defaultModel = models.find((item) => item.default)?.id ?? models[0]?.id ?? "";
        setSelectedAnswerModel(defaultModel);
      })
      .catch((modelError) => {
        if (active) {
          setError(modelError instanceof Error ? modelError.message : "Unable to load answer models");
        }
      });

    return () => {
      active = false;
    };
  }, [mode, answerModels.length]);

  async function handleSubmit() {
    if (!query.trim()) {
      return;
    }
    setIsLoading(true);
    setError(null);

    try {
      if (mode === "search") {
        const response = await search(
          query.trim(),
          Number.parseInt(searchLimit, 10),
          Number.parseFloat(searchMinScore),
        );
        setSearchResults(response.results);
      } else if (mode === "retrieve") {
        const response = await retrieve(query.trim());
        setRetrieveResults(response.retrieval_results);
      } else {
        setAnswerText("");
        setRetrieveResults([]);
        await streamAnswer({
          query: query.trim(),
          model: selectedAnswerModel,
          onAnswerDelta: (delta) => setAnswerText((current) => current + delta),
          onResults: (results) => setRetrieveResults(results),
        });
      }
    } catch (submissionError) {
      setError(submissionError instanceof Error ? submissionError.message : "Unknown request error");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleFormSubmit(event?: React.FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    await handleSubmit();
  }

  async function handleView(sourceId: string) {
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const response = await getSource(sourceId);
      setPreviewSource(response);
    } catch (sourceError) {
      setPreviewSource(null);
      setPreviewError(sourceError instanceof Error ? sourceError.message : "Unable to load source");
    } finally {
      setPreviewLoading(false);
    }
  }

  function closePreview() {
    setPreviewSource(null);
    setPreviewError(null);
    setPreviewLoading(false);
  }

  function clearQueryAndResults() {
    setQuery("");
    setSearchResults([]);
    setRetrieveResults([]);
    setAnswerText("");
    setError(null);
    closePreview();
  }

  async function copyResults() {
    const payload = mode === "search" ? searchResults : retrieveResults;
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    setResultsCopied(true);
    window.setTimeout(() => setResultsCopied(false), 1200);
  }

  async function copyChunk(chunk: string) {
    await navigator.clipboard.writeText(chunk);
  }

  const currentResultsCount = mode === "search" ? searchResults.length : retrieveResults.length;

  return (
    <div className="app-shell">
      <div className="main-shell">
        <header className="topbar">
          <div>
            <p className="eyebrow">Knowledge graph retrieval</p>
            <div className="topbar__title-row">
              <h1>estebanf&apos;s RAG</h1>
            </div>
          </div>
        </header>

        <main className="content">
          <form className="query-panel" onSubmit={handleFormSubmit}>
            <div className="tabs" role="tablist" aria-label="Mode selection">
              <button
                aria-selected={mode === "search"}
                className={`tab${mode === "search" ? " tab--active" : ""}`}
                role="tab"
                type="button"
                onClick={() => setMode("search")}
              >
                Search
              </button>
              <button
                aria-selected={mode === "retrieve"}
                className={`tab${mode === "retrieve" ? " tab--active" : ""}`}
                role="tab"
                type="button"
                onClick={() => setMode("retrieve")}
              >
                Retrieve
              </button>
              <button
                aria-selected={mode === "answer"}
                className={`tab${mode === "answer" ? " tab--active" : ""}`}
                role="tab"
                type="button"
                onClick={() => setMode("answer")}
              >
                Answer
              </button>
            </div>

            <label className="query-panel__label" htmlFor="semantic-query">
              Semantic Query
            </label>
            {mode === "search" ? (
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
                    value={searchMinScore}
                    onChange={(event) => setSearchMinScore(event.target.value)}
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
                    value={searchLimit}
                    onChange={(event) => setSearchLimit(event.target.value)}
                  />
                </label>
              </div>
            ) : null}
            {mode === "answer" ? (
              <div className="search-controls">
                <label className="search-controls__field" htmlFor="answer-model">
                  <span>Answer Model</span>
                  <select
                    aria-label="Answer Model"
                    id="answer-model"
                    value={selectedAnswerModel}
                    onChange={(event) => setSelectedAnswerModel(event.target.value)}
                  >
                    {answerModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            ) : null}
            <div className="query-panel__controls">
              <input
                id="semantic-query"
                placeholder="Search across documents and research models..."
                type="text"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              <div className="query-panel__actions">
                <button className="secondary-button" type="button" onClick={clearQueryAndResults}>
                  Clear
                </button>
                <button className="primary-button" type="submit">
                  {mode === "search" ? "Search" : mode === "retrieve" ? "Retrieve" : "Answer"}
                </button>
              </div>
            </div>
          </form>

          {mode === "answer" ? (
            <section className="answer-panel">
              <div className="answer-panel__header">
                <h3>Answer</h3>
              </div>
              <div className="answer-panel__body">
                {answerText ? answerText : isLoading ? "Thinking..." : "Run Answer to generate a grounded response."}
              </div>
            </section>
          ) : null}

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
              <span>
                {isLoading ? "Loading..." : `${currentResultsCount} result${currentResultsCount === 1 ? "" : "s"}`}
              </span>
            </div>

            {error ? <p className="panel-state panel-state--error">{error}</p> : null}

            {mode === "search" ? (
              <div className="results-stack">
                {searchResults.map((result) => (
                  <ResultCard key={result.chunk_id} result={result} onCopyChunk={copyChunk} onView={handleView} />
                ))}
              </div>
            ) : (
              <div className="results-stack">
                {retrieveResults.map((result) => (
                  <section className="retrieve-group" key={result.chunk_id}>
                    <ResultCard result={result} onCopyChunk={copyChunk} onView={handleView} />
                    {result.related.map((related) => (
                      <div className="related-group" key={`${result.chunk_id}-${related.entity}`}>
                        <div className="related-group__label">{related.entity}</div>
                        <div className="related-group__stack">
                          {related.chunks.map((chunk) => (
                            <ResultCard compact key={chunk.chunk_id} result={chunk} onCopyChunk={copyChunk} onView={handleView} />
                          ))}
                        </div>
                      </div>
                    ))}
                  </section>
                ))}
              </div>
            )}
          </section>
        </main>
      </div>

      <SourcePanel
        error={previewError}
        loading={previewLoading}
        source={previewSource}
        onClose={closePreview}
      />
    </div>
  );
}
