import { useEffect, useState } from "react";
import MonacoEditor from "@monaco-editor/react";

import type {
  AnswerModel,
  CommunityResponse,
  CommunityRunRequest,
  CommunityRunSummary,
  StageEntry,
  WorkingSet,
} from "../lib/api";
import {
  getAnswerModels,
  getCommunityRun,
  listCommunityRuns,
  listWorkingSets,
  startCommunityRun,
  streamRunEvents,
} from "../lib/api";
import argHelp from "../lib/argHelp";

type CommunityViewProps = {
  onView: (sourceId: string) => void;
  initialSourceIds?: string[];
};

export default function CommunityView({ onView, initialSourceIds }: CommunityViewProps) {
  const scopeInit = initialSourceIds && initialSourceIds.length > 0 ? "ids" : "working_set";
  const idsInit = initialSourceIds?.join("\n") || "";

  const [scopeMode, setScopeMode] = useState<"working_set" | "ids" | "search" | "retrieve">(
    scopeInit as "working_set" | "ids" | "search" | "retrieve",
  );
  const [workingSetId, setWorkingSetId] = useState("");
  const [explicitIds, setExplicitIds] = useState(idsInit);
  const [criteria, setCriteria] = useState("");
  const [searchLimit, setSearchLimit] = useState("10");
  const [searchMinScore, setSearchMinScore] = useState("0.7");
  const [retrieveSeedCount, setRetrieveSeedCount] = useState("10");
  const [retrieveResultCount, setRetrieveResultCount] = useState("5");
  const [retrieveRrfK, setRetrieveRrfK] = useState("60");

  const [resolution, setResolution] = useState("1.0");
  const [minCommunitySize, setMinCommunitySize] = useState("3");
  const [summarize, setSummarize] = useState(false);
  const [summarizeModel, setSummarizeModel] = useState("");
  const [answerModels, setAnswerModels] = useState<AnswerModel[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [semanticThreshold, setSemanticThreshold] = useState("0.85");
  const [sourceCoocWeight, setSourceCoocWeight] = useState("0.5");
  const [crossSourceTopK, setCrossSourceTopK] = useState("5");
  const [maxCrossSourceQueries, setMaxCrossSourceQueries] = useState("50");
  const [cutoff, setCutoff] = useState("0.5");
  const [topKChunks, setTopKChunks] = useState("5");

  const [workingSets, setWorkingSets] = useState<WorkingSet[]>([]);

  const [runId, setRunId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<"idle" | "running" | "completed" | "failed" | "connection_lost">("idle");
  const [stageEntries, setStageEntries] = useState<StageEntry[]>([]);
  const [result, setResult] = useState<CommunityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const [runs, setRuns] = useState<CommunityRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getAnswerModels()
      .then((models) => {
        if (!active) return;
        setAnswerModels(models ?? []);
        const defaultModel = (models ?? []).find((item) => item.default)?.id ?? (models ?? [])[0]?.id ?? "";
        setSummarizeModel((prev) => prev || defaultModel);
      })
      .catch(() => {});
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    listWorkingSets()
      .then((resp) => { if (active) setWorkingSets(resp.working_sets ?? []); })
      .catch(() => { if (active) setWorkingSets([]); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    setRunsLoading(true);
    listCommunityRuns()
      .then((resp) => { if (active) setRuns(resp.runs ?? []); })
      .catch(() => { if (active) setRuns([]); })
      .finally(() => { if (active) setRunsLoading(false); });
    return () => { active = false; };
  }, [runStatus]);

  useEffect(() => {
    if (!runs.length || runStatus !== "idle") return;
    const latest = runs[0];
    if (latest && latest.status === "running") {
      setRunId(latest.run_id);
      setRunStatus("running");
      attachToStream(latest.run_id);
    }
  }, [runs, runStatus]);

  async function attachToStream(id: string) {
    const controller = new AbortController();
    setAbortController(controller);
    setStageEntries([]);
    setError(null);

    try {
      await streamRunEvents(
        id,
        {
          onStage: (entry) => {
            setStageEntries((prev) => [...prev, entry]);
          },
          onResult: (data) => {
            setResult(data);
            setRunStatus("completed");
          },
          onError: (err) => {
            setError(err.message || "Run failed");
            setRunStatus("failed");
          },
        },
        controller.signal,
      );
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        return;
      }
      setRunStatus("connection_lost");
      setError("Connection lost — reconnecting");
    }
  }

  async function handleRun() {
    const body: CommunityRunRequest = { scope_mode: scopeMode };

    if (scopeMode === "working_set") {
      if (!workingSetId) return;
      body.working_set_id = workingSetId;
    } else if (scopeMode === "ids") {
      const ids = explicitIds.split("\n").map((s) => s.trim()).filter(Boolean);
      if (!ids.length) return;
      body.source_ids = ids;
    } else {
      const lines = criteria.split("\n").map((s) => s.trim()).filter(Boolean);
      if (!lines.length) return;
      body.criteria = lines;
      if (scopeMode === "search") {
        body.search_options = {
          limit: Number.parseInt(searchLimit, 10),
          min_score: Number.parseFloat(searchMinScore),
        };
      }
      if (scopeMode === "retrieve") {
        body.retrieve_options = {
          ...(retrieveSeedCount ? { seed_count: Number.parseInt(retrieveSeedCount, 10) } : {}),
          ...(retrieveResultCount ? { result_count: Number.parseInt(retrieveResultCount, 10) } : {}),
          ...(retrieveRrfK ? { rrf_k: Number.parseInt(retrieveRrfK, 10) } : {}),
        };
      }
    }

    if (resolution) body.resolution = Number.parseFloat(resolution);
    if (semanticThreshold) body.semantic_threshold = Number.parseFloat(semanticThreshold);
    if (sourceCoocWeight) body.source_cooc_weight = Number.parseFloat(sourceCoocWeight);
    if (crossSourceTopK) body.cross_source_top_k = Number.parseInt(crossSourceTopK, 10);
    if (maxCrossSourceQueries) body.max_cross_source_queries = Number.parseInt(maxCrossSourceQueries, 10);
    if (cutoff) body.cutoff = Number.parseFloat(cutoff);
    if (minCommunitySize) body.min_community_size = Number.parseInt(minCommunitySize, 10);
    if (topKChunks) body.top_k_chunks = Number.parseInt(topKChunks, 10);
    if (summarize && summarizeModel) {
      body.summarize = true;
      body.summarize_model = summarizeModel;
    }

    setError(null);
    setRunStatus("running");
    setStageEntries([]);
    setResult(null);

    try {
      const resp = await startCommunityRun(body);
      const newRunId = resp.run_id;
      setRunId(newRunId);
      attachToStream(newRunId);
    } catch (e) {
      setRunStatus("failed");
      setError(e instanceof Error ? e.message : "Failed to start run");
    }
  }

  function handleCancel() {
    if (abortController) {
      abortController.abort();
    }
    setRunStatus("idle");
    setStageEntries((prev) => [...prev, { stage: "cancelled", message: "Run continues in background — see run history" }]);
  }

  function handleRetry() {
    setRunStatus("idle");
    setStageEntries([]);
    setError(null);
  }

  async function handleLoadRun(runSummary: CommunityRunSummary) {
    try {
      const detail = await getCommunityRun(runSummary.run_id);
      setSelectedRunId(runSummary.run_id);
      if (detail.status === "completed" && detail.result) {
        setResult(detail.result);
        setRunStatus("completed");
        setStageEntries([]);
        setRunId(runSummary.run_id);
      } else if (detail.status === "failed") {
        setRunStatus("failed");
        setError(detail.error || "Run failed");
        setRunId(runSummary.run_id);
        setStageEntries([]);
      } else if (detail.status === "running") {
        setRunId(runSummary.run_id);
        setRunStatus("running");
        attachToStream(runSummary.run_id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load run");
    }
  }

  const communityCount = result?.communities?.length ?? 0;
  const virtualMerges = result?.virtual_merges;

  return (
    <div className="content">
      <form
        className="query-panel"
        aria-label="Community form"
        onSubmit={(e) => {
          e.preventDefault();
          handleRun();
        }}
      >
        <div className="community-form">
          <label className="search-controls__field community-scope-row" htmlFor="community-scope">
            <span>Scope Mode</span>
            <select
              id="community-scope"
              value={scopeMode}
              onChange={(e) => setScopeMode(e.target.value as "working_set" | "ids" | "search" | "retrieve")}
            >
              <option value="working_set">Working Set</option>
              <option value="ids">IDs</option>
              <option value="search">Search</option>
              <option value="retrieve">Retrieve</option>
            </select>
          </label>

          {scopeMode === "working_set" ? (
            <label className="search-controls__field" htmlFor="working-set-select">
              <span>Working Set</span>
              <select
                id="working-set-select"
                value={workingSetId}
                onChange={(e) => setWorkingSetId(e.target.value)}
              >
                <option value="">Select a working set...</option>
                {workingSets.map((ws) => (
                  <option key={ws.id} value={ws.id}>
                    {ws.name} ({ws.source_count} sources)
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          {scopeMode === "ids" ? (
            <>
              <label className="query-panel__label" htmlFor="community-input">
                Source IDs (one per line)
              </label>
              <textarea
                className="community-textarea"
                id="community-input"
                placeholder="uuid-1\nuuid-2"
                rows={4}
                value={explicitIds}
                onChange={(e) => setExplicitIds(e.target.value)}
              />
            </>
          ) : null}

          {scopeMode === "search" || scopeMode === "retrieve" ? (
            <>
              <label className="query-panel__label" htmlFor="community-input">
                Criteria (one per line)
              </label>
              <textarea
                className="community-textarea"
                id="community-input"
                placeholder="machine learning\nneural networks"
                rows={4}
                value={criteria}
                onChange={(e) => setCriteria(e.target.value)}
              />
            </>
          ) : null}

          {scopeMode === "search" ? (
            <div className="search-controls">
              <label className="search-controls__field" htmlFor="comm-search-limit">
                <span>Limit</span>
                <input id="comm-search-limit" min="1" step="1" type="number" value={searchLimit} onChange={(e) => setSearchLimit(e.target.value)} />
              </label>
              <label className="search-controls__field" htmlFor="comm-search-min-score">
                <span>Min Score</span>
                <input id="comm-search-min-score" max="1" min="0" step="0.01" type="number" value={searchMinScore} onChange={(e) => setSearchMinScore(e.target.value)} />
              </label>
            </div>
          ) : null}

          {scopeMode === "retrieve" ? (
            <div className="search-controls community-retrieve-controls">
              <label className="search-controls__field" htmlFor="comm-seed-count">
                <span>Seed Count</span>
                <input id="comm-seed-count" min="1" step="1" type="number" value={retrieveSeedCount} onChange={(e) => setRetrieveSeedCount(e.target.value)} />
              </label>
              <label className="search-controls__field" htmlFor="comm-result-count">
                <span>Result Count</span>
                <input id="comm-result-count" min="1" step="1" type="number" value={retrieveResultCount} onChange={(e) => setRetrieveResultCount(e.target.value)} />
              </label>
              <label className="search-controls__field" htmlFor="comm-rrf-k">
                <span>RRF K</span>
                <input id="comm-rrf-k" min="1" step="1" type="number" value={retrieveRrfK} onChange={(e) => setRetrieveRrfK(e.target.value)} />
              </label>
            </div>
          ) : null}

          <div className="search-controls community-options-grid">
            <label className="search-controls__field" htmlFor="comm-resolution">
              <span>Resolution</span>
              <input id="comm-resolution" min="0.1" step="0.1" type="number" value={resolution} onChange={(e) => setResolution(e.target.value)} />
              {argHelp.resolution ? <span className="arg-help">{argHelp.resolution}</span> : null}
            </label>
            <label className="search-controls__field" htmlFor="comm-min-size">
              <span>Min Community Size</span>
              <input id="comm-min-size" min="1" step="1" type="number" value={minCommunitySize} onChange={(e) => setMinCommunitySize(e.target.value)} />
              {argHelp.min_community_size ? <span className="arg-help">{argHelp.min_community_size}</span> : null}
            </label>
          </div>

          <div className="community-summarize-row">
            <label className="community-summarize-label">
              <input
                checked={summarize}
                type="checkbox"
                onChange={(e) => setSummarize(e.target.checked)}
              />
              <span>Enable Summarization</span>
            </label>
            {summarize ? (
              <label className="search-controls__field" htmlFor="community-model">
                <span>Summarize Model</span>
                <select
                  id="community-model"
                  value={summarizeModel}
                  onChange={(e) => setSummarizeModel(e.target.value)}
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
              <label className="search-controls__field" htmlFor="comm-sem-threshold">
                <span>Semantic Threshold</span>
                <input id="comm-sem-threshold" max="1" min="0" step="0.01" type="number" value={semanticThreshold} onChange={(e) => setSemanticThreshold(e.target.value)} />
                {argHelp.semantic_threshold ? <span className="arg-help">{argHelp.semantic_threshold}</span> : null}
              </label>
              <label className="search-controls__field" htmlFor="comm-source-cooc">
                <span>Source Cooc Weight</span>
                <input id="comm-source-cooc" max="1" min="0" step="0.01" type="number" value={sourceCoocWeight} onChange={(e) => setSourceCoocWeight(e.target.value)} />
                {argHelp.source_cooc_weight ? <span className="arg-help">{argHelp.source_cooc_weight}</span> : null}
              </label>
              <label className="search-controls__field" htmlFor="comm-cross-topk">
                <span>Cross-source Top K</span>
                <input id="comm-cross-topk" min="1" step="1" type="number" value={crossSourceTopK} onChange={(e) => setCrossSourceTopK(e.target.value)} />
                {argHelp.cross_source_top_k ? <span className="arg-help">{argHelp.cross_source_top_k}</span> : null}
              </label>
              <label className="search-controls__field" htmlFor="comm-max-cross">
                <span>Max Cross-source Queries</span>
                <input id="comm-max-cross" min="1" step="1" type="number" value={maxCrossSourceQueries} onChange={(e) => setMaxCrossSourceQueries(e.target.value)} />
                {argHelp.max_cross_source_queries ? <span className="arg-help">{argHelp.max_cross_source_queries}</span> : null}
              </label>
              <label className="search-controls__field" htmlFor="comm-cutoff">
                <span>Cutoff</span>
                <input id="comm-cutoff" max="1" min="0" step="0.01" type="number" value={cutoff} onChange={(e) => setCutoff(e.target.value)} />
                {argHelp.cutoff ? <span className="arg-help">{argHelp.cutoff}</span> : null}
              </label>
              <label className="search-controls__field" htmlFor="comm-top-k">
                <span>Top K Chunks</span>
                <input id="comm-top-k" min="1" step="1" type="number" value={topKChunks} onChange={(e) => setTopKChunks(e.target.value)} />
                {argHelp.top_k_chunks ? <span className="arg-help">{argHelp.top_k_chunks}</span> : null}
              </label>
            </div>
          ) : null}

          <div className="community-actions">
            {(runStatus === "running") ? (
              <button className="secondary-button" type="button" onClick={handleCancel}>
                Cancel
              </button>
            ) : null}
            {(runStatus === "failed" || runStatus === "connection_lost") ? (
              <button className="secondary-button" type="button" onClick={handleRetry}>
                Retry
              </button>
            ) : null}
            <button className="primary-button" type="submit" disabled={runStatus === "running"}>
              {runStatus === "running" ? "Running..." : "Run"}
            </button>
          </div>
        </div>
      </form>

      {runStatus === "running" || stageEntries.length > 0 || result || error ? (
        <section className="community-panel">
          <div className="results-panel__header">
            <div className="results-panel__title-row">
              <h3>Progress</h3>
            </div>
            <span>
              {runStatus === "running"
                ? "Running..."
                : runStatus === "completed"
                  ? `${communityCount} communit${communityCount === 1 ? "y" : "ies"}`
                  : runStatus === "connection_lost"
                    ? "Connection lost"
                    : ""}
            </span>
          </div>

          {error ? <p className="panel-state panel-state--error">{error}</p> : null}

          {stageEntries.length > 0 ? (
            <div className="stage-progress">
              {stageEntries.map((entry, i) => (
                <p className="stage-entry" key={`${entry.stage}-${i}`}>
                  <span className="stage-entry__stage">{entry.stage}</span> {entry.message}
                </p>
              ))}
            </div>
          ) : null}

          {runStatus === "connection_lost" ? (
            <p className="panel-state">Connection to run events was lost. The run may still be active — check run history.</p>
          ) : null}

          {summarize && result && result.communities.length > 0 ? (
            <div className="community-cards">
              {result.communities.map((c) => (
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

          {virtualMerges !== undefined ? (
            <p className="panel-state">Virtual merges: {virtualMerges}</p>
          ) : null}

          <div className="community-json">
            <MonacoEditor
              height="480px"
              language="json"
              options={{ readOnly: true, minimap: { enabled: false }, scrollBeyondLastLine: false, fontSize: 13 }}
              theme="vs"
              value={result ? JSON.stringify(result, null, 2) : "// Run to see results"}
            />
          </div>
        </section>
      ) : null}

      <section className="community-panel">
        <div className="results-panel__header">
          <h3>Run History</h3>
        </div>
        {runsLoading ? <p className="panel-state">Loading runs...</p> : null}
        {!runsLoading && runs.length === 0 ? <p className="panel-state">No past runs.</p> : null}
        {runs.length > 0 ? (
          <div className="run-history-list">
            {runs.map((run) => (
              <div
                className={`run-history-item${selectedRunId === run.run_id ? " run-history-item--selected" : ""}`}
                key={run.run_id}
              >
                <div className="run-history-item__info">
                  <span className={`badge run-status-badge run-status-badge--${run.status}`}>
                    {run.status}
                  </span>
                  <span className="run-history-item__id">{run.run_id.slice(0, 8)}</span>
                  <span className="run-history-item__date">
                    {new Date(run.created_at).toLocaleString()}
                  </span>
                </div>
                <button className="ghost-button" type="button" onClick={() => handleLoadRun(run)}>
                  Load
                </button>
              </div>
            ))}
          </div>
        ) : null}
      </section>
    </div>
  );
}
