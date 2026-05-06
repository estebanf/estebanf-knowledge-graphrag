import { useEffect, useState } from "react";
import MonacoEditor from "@monaco-editor/react";

import BucketPopover from "./components/BucketPopover";
import InsightCard from "./components/InsightCard";
import ResultCard from "./components/ResultCard";
import SourcePanel from "./components/SourcePanel";
import SourcesExplorer from "./components/SourcesExplorer";
import type { AnswerModel, CommunityRequestOptions, CommunityResponse, MetadataFilter, RetrieveResponse, SearchResponse, SourceDetail, SourceInsight, SourceSummary } from "./lib/api";
import { community, getAnswerModels, getSource, getSourceInsights, listSources, retrieve, search, streamAnswer } from "./lib/api";

type Mode = "search" | "retrieve" | "answer" | "community" | "sources";
type BucketEntry = { sourceId: string; title: string };
const SOURCES_PAGE_SIZE = 20;

export default function App() {
  const [mode, setMode] = useState<Mode>("search");
  const [query, setQuery] = useState("");
  const [searchMinScore, setSearchMinScore] = useState("0.7");
  const [searchLimit, setSearchLimit] = useState("10");
  const [searchResults, setSearchResults] = useState<SearchResponse["results"]>({ chunks: [], insights: [] });
  const [retrieveResults, setRetrieveResults] = useState<RetrieveResponse["retrieval_results"]>([]);
  const [retrieveInsights, setRetrieveInsights] = useState<RetrieveResponse["insights"]>([]);
  const [answerText, setAnswerText] = useState("");
  const [answerModels, setAnswerModels] = useState<AnswerModel[]>([]);
  const [selectedAnswerModel, setSelectedAnswerModel] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewSource, setPreviewSource] = useState<SourceDetail | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [resultsCopied, setResultsCopied] = useState(false);
  const [bucket, setBucket] = useState<BucketEntry[]>([]);
  const [sources, setSources] = useState<SourceSummary[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [sourcesOffset, setSourcesOffset] = useState(0);
  const [sourcesTotal, setSourcesTotal] = useState(0);
  const [sourceMetadataFilters, setSourceMetadataFilters] = useState<MetadataFilter[]>([]);
  const [selectedSource, setSelectedSource] = useState<SourceDetail | null>(null);
  const [sourceInsights, setSourceInsights] = useState<SourceInsight[]>([]);
  const [sourcesLoading, setSourcesLoading] = useState(false);
  const [selectedSourceLoading, setSelectedSourceLoading] = useState(false);
  const [sourceInsightsLoading, setSourceInsightsLoading] = useState(false);
  const [sourcesError, setSourcesError] = useState<string | null>(null);
  const [selectedSourceError, setSelectedSourceError] = useState<string | null>(null);
  const [sourceInsightsError, setSourceInsightsError] = useState<string | null>(null);

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

  // Community tab state
  const [communityScopeMode, setCommunityScopeMode] = useState<"ids" | "search" | "retrieve">("ids");
  const [communityInput, setCommunityInput] = useState("");
  const [communitySearchLimit, setCommunitySearchLimit] = useState("10");
  const [communitySearchMinScore, setCommunitySearchMinScore] = useState("0.7");
  const [communityRetrieveSeedCount, setCommunityRetrieveSeedCount] = useState("10");
  const [communityRetrieveResultCount, setCommunityRetrieveResultCount] = useState("5");
  const [communityRetrieveRrfK, setCommunityRetrieveRrfK] = useState("60");
  const [communitySemanticThreshold, setCommunitySemanticThreshold] = useState("0.85");
  const [communityCutoff, setCommunityCutoff] = useState("0.5");
  const [communityMinSize, setCommunityMinSize] = useState("3");
  const [communityTopK, setCommunityTopK] = useState("5");
  const [communitySummarize, setCommunitySummarize] = useState(false);
  const [communityModel, setCommunityModel] = useState("");
  const [communityResult, setCommunityResult] = useState<CommunityResponse | null>(null);

  useEffect(() => {
    if ((mode !== "answer" && mode !== "community") || answerModels.length > 0) {
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
        setCommunityModel((prev) => prev || defaultModel);
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

  useEffect(() => {
    if (mode !== "sources") {
      return;
    }

    let active = true;
    setSourcesLoading(true);
    setSourcesError(null);
    listSources(SOURCES_PAGE_SIZE, sourcesOffset, sourceMetadataFilters)
      .then((response) => {
        if (!active) {
          return;
        }
        setSources(response.sources);
        setSourcesTotal(response.total);
        setSelectedSourceId(response.sources[0]?.source_id || null);
      })
      .catch((sourceListError) => {
        if (active) {
          setSourcesError(sourceListError instanceof Error ? sourceListError.message : "Unable to load sources");
        }
      })
      .finally(() => {
        if (active) {
          setSourcesLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [mode, sourcesOffset, sourceMetadataFilters]);

  useEffect(() => {
    if (mode !== "sources" || !selectedSourceId) {
      return;
    }

    let active = true;
    setSelectedSourceLoading(true);
    setSourceInsightsLoading(true);
    setSelectedSourceError(null);
    setSourceInsightsError(null);
    getSource(selectedSourceId)
      .then((response) => {
        if (active) {
          setSelectedSource(response);
        }
      })
      .catch((sourceError) => {
        if (active) {
          setSelectedSource(null);
          setSelectedSourceError(sourceError instanceof Error ? sourceError.message : "Unable to load source");
        }
      })
      .finally(() => {
        if (active) {
          setSelectedSourceLoading(false);
        }
      });

    getSourceInsights(selectedSourceId)
      .then((response) => {
        if (active) {
          setSourceInsights(response.insights);
        }
      })
      .catch((insightsError) => {
        if (active) {
          setSourceInsights([]);
          setSourceInsightsError(insightsError instanceof Error ? insightsError.message : "Unable to load insights");
        }
      })
      .finally(() => {
        if (active) {
          setSourceInsightsLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [mode, selectedSourceId]);

  async function handleSubmit() {
    if (mode === "sources") {
      return;
    }
    if (mode !== "community" && !query.trim()) {
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
        setRetrieveInsights(response.insights || []);
      } else if (mode === "community") {
        const lines = communityInput.split("\n").map((s) => s.trim()).filter(Boolean);
        const body: CommunityRequestOptions = { scope_mode: communityScopeMode };
        if (communityScopeMode === "ids") {
          body.source_ids = lines;
        } else {
          body.criteria = lines;
        }
        if (communityScopeMode === "search") {
          body.search_options = {
            limit: Number.parseInt(communitySearchLimit, 10),
            min_score: Number.parseFloat(communitySearchMinScore),
          };
        }
        if (communityScopeMode === "retrieve") {
          body.retrieve_options = {
            ...(communityRetrieveSeedCount ? { seed_count: Number.parseInt(communityRetrieveSeedCount, 10) } : {}),
            ...(communityRetrieveResultCount ? { result_count: Number.parseInt(communityRetrieveResultCount, 10) } : {}),
            ...(communityRetrieveRrfK ? { rrf_k: Number.parseInt(communityRetrieveRrfK, 10) } : {}),
          };
        }
        const communityOpts: CommunityRequestOptions["community_options"] = {};
        if (communitySemanticThreshold) communityOpts.semantic_threshold = Number.parseFloat(communitySemanticThreshold);
        if (communityCutoff) communityOpts.cutoff = Number.parseFloat(communityCutoff);
        if (communityMinSize) communityOpts.min_community_size = Number.parseInt(communityMinSize, 10);
        if (communityTopK) communityOpts.top_k_chunks = Number.parseInt(communityTopK, 10);
        if (Object.keys(communityOpts).length > 0) body.community_options = communityOpts;
        if (communitySummarize && communityModel) body.summarize_model = communityModel;
        const response = await community(body);
        setCommunityResult(response);
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
    setSearchResults({ chunks: [], insights: [] });
    setRetrieveResults([]);
    setAnswerText("");
    setCommunityResult(null);
    setCommunityInput("");
    setError(null);
    closePreview();
  }

  async function copyResults() {
    const payload = mode === "search" ? searchResults : mode === "community" ? communityResult : retrieveResults;
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    setResultsCopied(true);
    window.setTimeout(() => setResultsCopied(false), 1200);
  }

  async function copyChunk(chunk: string) {
    await navigator.clipboard.writeText(chunk);
  }

  function handleNextSourcesPage() {
    setSourcesOffset((current) => current + SOURCES_PAGE_SIZE);
  }

  function handlePreviousSourcesPage() {
    setSourcesOffset((current) => Math.max(0, current - SOURCES_PAGE_SIZE));
  }

  function handleToggleSourceMetadataFilter(filter: MetadataFilter) {
    setSourcesOffset(0);
    setSourceMetadataFilters((current) => {
      if (current.some((item) => item.key === filter.key && item.value === filter.value)) {
        return current;
      }
      return [...current, filter];
    });
  }

  function handleRemoveSourceMetadataFilter(filter: MetadataFilter) {
    setSourcesOffset(0);
    setSourceMetadataFilters((current) => current.filter((item) => item.key !== filter.key || item.value !== filter.value));
  }

  const currentResultsCount = mode === "search" ? searchResults.chunks.length + searchResults.insights.length : retrieveResults.length;
  const sourcesPage = Math.floor(sourcesOffset / SOURCES_PAGE_SIZE) + 1;
  const sourcesPageCount = Math.max(1, Math.ceil(sourcesTotal / SOURCES_PAGE_SIZE));

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
          <BucketPopover
            entries={bucket}
            onClear={handleClearBucket}
            onCopy={handleCopyBucket}
          />
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
              <button
                aria-selected={mode === "community"}
                className={`tab${mode === "community" ? " tab--active" : ""}`}
                role="tab"
                type="button"
                onClick={() => setMode("community")}
              >
                Community
              </button>
              <button
                aria-selected={mode === "sources"}
                className={`tab${mode === "sources" ? " tab--active" : ""}`}
                role="tab"
                type="button"
                onClick={() => setMode("sources")}
              >
                Sources
              </button>
            </div>

            {mode !== "community" && mode !== "sources" ? (
              <label className="query-panel__label" htmlFor="semantic-query">
                Semantic Query
              </label>
            ) : null}
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
            {mode === "sources" ? null : mode === "community" ? (
              <div className="community-form">
                <label className="search-controls__field community-scope-row" htmlFor="community-scope">
                  <span>Scope Mode</span>
                  <select
                    id="community-scope"
                    value={communityScopeMode}
                    onChange={(e) => setCommunityScopeMode(e.target.value as "ids" | "search" | "retrieve")}
                  >
                    <option value="ids">IDs</option>
                    <option value="search">Search</option>
                    <option value="retrieve">Retrieve</option>
                  </select>
                </label>
                <label className="query-panel__label" htmlFor="community-input">
                  {communityScopeMode === "ids" ? "Source IDs (one per line)" : "Criteria (one per line)"}
                </label>
                <textarea
                  className="community-textarea"
                  id="community-input"
                  placeholder={communityScopeMode === "ids" ? "uuid-1\nuuid-2" : "machine learning\nneural networks"}
                  rows={4}
                  value={communityInput}
                  onChange={(e) => setCommunityInput(e.target.value)}
                />
                {communityScopeMode === "search" ? (
                  <div className="search-controls">
                    <label className="search-controls__field" htmlFor="comm-search-limit">
                      <span>Limit</span>
                      <input id="comm-search-limit" min="1" step="1" type="number" value={communitySearchLimit} onChange={(e) => setCommunitySearchLimit(e.target.value)} />
                    </label>
                    <label className="search-controls__field" htmlFor="comm-search-min-score">
                      <span>Min Score</span>
                      <input id="comm-search-min-score" max="1" min="0" step="0.01" type="number" value={communitySearchMinScore} onChange={(e) => setCommunitySearchMinScore(e.target.value)} />
                    </label>
                  </div>
                ) : null}
                {communityScopeMode === "retrieve" ? (
                  <div className="search-controls community-retrieve-controls">
                    <label className="search-controls__field" htmlFor="comm-seed-count">
                      <span>Seed Count</span>
                      <input id="comm-seed-count" min="1" step="1" type="number" value={communityRetrieveSeedCount} onChange={(e) => setCommunityRetrieveSeedCount(e.target.value)} />
                    </label>
                    <label className="search-controls__field" htmlFor="comm-result-count">
                      <span>Result Count</span>
                      <input id="comm-result-count" min="1" step="1" type="number" value={communityRetrieveResultCount} onChange={(e) => setCommunityRetrieveResultCount(e.target.value)} />
                    </label>
                    <label className="search-controls__field" htmlFor="comm-rrf-k">
                      <span>RRF K</span>
                      <input id="comm-rrf-k" min="1" step="1" type="number" value={communityRetrieveRrfK} onChange={(e) => setCommunityRetrieveRrfK(e.target.value)} />
                    </label>
                  </div>
                ) : null}
                <div className="search-controls community-options-grid">
                  <label className="search-controls__field" htmlFor="comm-sem-threshold">
                    <span>Semantic Threshold</span>
                    <input id="comm-sem-threshold" max="1" min="0" step="0.01" type="number" value={communitySemanticThreshold} onChange={(e) => setCommunitySemanticThreshold(e.target.value)} />
                  </label>
                  <label className="search-controls__field" htmlFor="comm-cutoff">
                    <span>Cutoff</span>
                    <input id="comm-cutoff" max="1" min="0" step="0.01" type="number" value={communityCutoff} onChange={(e) => setCommunityCutoff(e.target.value)} />
                  </label>
                  <label className="search-controls__field" htmlFor="comm-min-size">
                    <span>Min Community Size</span>
                    <input id="comm-min-size" min="1" step="1" type="number" value={communityMinSize} onChange={(e) => setCommunityMinSize(e.target.value)} />
                  </label>
                  <label className="search-controls__field" htmlFor="comm-top-k">
                    <span>Top K Chunks</span>
                    <input id="comm-top-k" min="1" step="1" type="number" value={communityTopK} onChange={(e) => setCommunityTopK(e.target.value)} />
                  </label>
                </div>
                <div className="community-summarize-row">
                  <label className="community-summarize-label">
                    <input
                      checked={communitySummarize}
                      type="checkbox"
                      onChange={(e) => setCommunitySummarize(e.target.checked)}
                    />
                    <span>Enable Summarization</span>
                  </label>
                  {communitySummarize ? (
                    <label className="search-controls__field" htmlFor="community-model">
                      <span>Summarize Model</span>
                      <select
                        id="community-model"
                        value={communityModel}
                        onChange={(e) => setCommunityModel(e.target.value)}
                      >
                        {answerModels.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : null}
                </div>
                <div className="query-panel__actions community-actions">
                  <button className="secondary-button" type="button" onClick={clearQueryAndResults}>
                    Clear
                  </button>
                  <button className="primary-button" type="submit">
                    Detect Communities
                  </button>
                </div>
              </div>
            ) : (
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
            )}
          </form>

          {mode === "sources" ? (
            <SourcesExplorer
              insights={sourceInsights}
              insightsError={sourceInsightsError}
              loadingInsights={sourceInsightsLoading}
              loadingSource={selectedSourceLoading}
              loadingSources={sourcesLoading}
              page={sourcesPage}
              pageCount={sourcesPageCount}
              selectedSourceId={selectedSourceId}
              selectedFilters={sourceMetadataFilters}
              source={selectedSource}
              sourceError={selectedSourceError}
              sources={sources}
              sourcesError={sourcesError}
              onNextPage={handleNextSourcesPage}
              onPreviousPage={handlePreviousSourcesPage}
              onRemoveMetadataFilter={handleRemoveSourceMetadataFilter}
              onSelectSource={setSelectedSourceId}
              onToggleMetadataFilter={handleToggleSourceMetadataFilter}
            />
          ) : mode === "answer" ? (
            <section className="answer-panel">
              <div className="answer-panel__header">
                <h3>Answer</h3>
              </div>
              <div className="answer-panel__body">
                {answerText ? answerText : isLoading ? "Thinking..." : "Run Answer to generate a grounded response."}
              </div>
            </section>
          ) : null}

          {mode === "sources" ? null : mode === "community" ? (
            <section className="community-panel">
              <div className="results-panel__header">
                <div className="results-panel__title-row">
                  <h3>Results</h3>
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
                  {isLoading
                    ? "Loading..."
                    : communityResult
                      ? `${communityResult.communities.length} communit${communityResult.communities.length === 1 ? "y" : "ies"}`
                      : "Run Detect Communities to see results"}
                </span>
              </div>

              {error ? <p className="panel-state panel-state--error">{error}</p> : null}

              {communitySummarize && communityResult && communityResult.communities.length > 0 ? (
                <div className="community-cards">
                  {communityResult.communities.map((c) => (
                    <div className="community-card" key={c.community_id}>
                      <div className="community-card__header">
                        <span className="badge">Community {c.community_id}</span>
                        {c.is_cross_source ? <span className="badge">Cross-source</span> : null}
                        <span className="score-chip">{c.entity_count} entities</span>
                      </div>
                      <div className="community-card__sources">
                        {c.contributing_sources.map((s) => s.source_name || s.source_id).join(", ")}
                      </div>
                      {c.summary ? (
                        <p className="community-card__summary">{c.summary}</p>
                      ) : (
                        <p className="community-card__summary community-card__summary--empty">No summary generated.</p>
                      )}
                    </div>
                  ))}
                </div>
              ) : null}

              <div className="community-json">
                <MonacoEditor
                  height="480px"
                  language="json"
                  options={{ readOnly: true, minimap: { enabled: false }, scrollBeyondLastLine: false, fontSize: 13 }}
                  theme="vs"
                  value={communityResult ? JSON.stringify(communityResult, null, 2) : "// Run Detect Communities to see results"}
                />
              </div>
            </section>
          ) : (
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
                <div className="results-sections">
                  {searchResults.insights.length > 0 ? (
                    <section className="results-section">
                      <h3 className="results-section__heading">Insights</h3>
                      <div className="results-stack">
                        {searchResults.insights.map((result) => (
                          <InsightCard key={result.insight_id} result={result} onCopy={copyChunk} onView={handleView} onAddToBucket={handleAddToBucket} />
                        ))}
                      </div>
                    </section>
                  ) : null}
                  {searchResults.chunks.length > 0 ? (
                    <section className="results-section">
                      <h3 className="results-section__heading">Chunks</h3>
                      <div className="results-stack">
                        {searchResults.chunks.map((result) => (
                          <ResultCard key={result.chunk_id} result={result} onCopyChunk={copyChunk} onView={handleView} onAddToBucket={handleAddToBucket} />
                        ))}
                      </div>
                    </section>
                  ) : null}
                  {searchResults.insights.length === 0 && searchResults.chunks.length === 0 ? (
                    <p className="panel-state">No results found.</p>
                  ) : null}
                </div>
              ) : (
                <div className="results-stack">
                  {retrieveInsights.length > 0 ? (
                    <section className="results-section">
                      <h3 className="results-section__heading">Insights</h3>
                      <div className="results-stack">
                        {retrieveInsights.map((seed) => (
                          <section className="retrieve-group" key={seed.insight_id}>
                            <InsightCard result={{...seed, topics: seed.topics || [], sources: seed.sources || []}} onCopy={copyChunk} onView={handleView} onAddToBucket={handleAddToBucket} />
                            {seed.related.map((group) => (
                              <div className="related-group" key={`${seed.insight_id}-${group.type}-${group.sub_query || "first"}`}>
                                <div className="related-group__label">
                                  {group.type === "first_hop" ? "Related" : `Related · ${group.sub_query}`}
                                </div>
                                <div className="related-group__stack">
                                  {group.insights.map((insight) => (
                                    <InsightCard key={insight.insight_id} result={{...insight, topics: insight.topics || [], sources: insight.sources || []}} onCopy={copyChunk} onView={handleView} onAddToBucket={handleAddToBucket} />
                                  ))}
                                </div>
                              </div>
                            ))}
                          </section>
                        ))}
                      </div>
                    </section>
                  ) : null}
                  {retrieveResults.map((result) => (
                    <section className="retrieve-group" key={result.chunk_id}>
                      <ResultCard result={result} onCopyChunk={copyChunk} onView={handleView} onAddToBucket={handleAddToBucket} />
                      {result.related.map((related) => (
                        <div className="related-group" key={`${result.chunk_id}-${related.entity}`}>
                          <div className="related-group__label">{related.entity}</div>
                          <div className="related-group__stack">
                            {related.chunks.map((chunk) => (
                              <ResultCard compact key={chunk.chunk_id} result={chunk} onCopyChunk={copyChunk} onView={handleView} onAddToBucket={handleAddToBucket} />
                            ))}
                          </div>
                        </div>
                      ))}
                    </section>
                  ))}
                </div>
              )}
            </section>
          )}
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
