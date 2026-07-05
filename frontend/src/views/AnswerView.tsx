import { useEffect, useState } from "react";
import type { AnswerModel, EvidenceItem, RetrieveResponse, WorkingSet } from "../lib/api";
import { getAnswerModels, listWorkingSets, saveAnswer, streamAnswer } from "../lib/api";
import ResultCard from "../components/ResultCard";

type AnswerViewProps = {
  onView: (sourceId: string) => void;
  onCopyChunk: (chunk: string) => Promise<void>;
};

export default function AnswerView({ onView, onCopyChunk }: AnswerViewProps) {
  const [query, setQuery] = useState("");
  const [answerText, setAnswerText] = useState("");
  const [answerModels, setAnswerModels] = useState<AnswerModel[]>([]);
  const [selectedAnswerModel, setSelectedAnswerModel] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retrieveResults, setRetrieveResults] = useState<RetrieveResponse["retrieval_results"]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [seedCount, setSeedCount] = useState("");
  const [resultCount, setResultCount] = useState("");
  const [rrfK, setRrfK] = useState("");

  const [workingSets, setWorkingSets] = useState<WorkingSet[]>([]);
  const [selectedWorkingSetId, setSelectedWorkingSetId] = useState("");

  const [savedId, setSavedId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let active = true;
    getAnswerModels()
      .then((models) => {
        if (!active) return;
        setAnswerModels(models ?? []);
        const defaultModel = (models ?? []).find((item) => item.default)?.id ?? models?.[0]?.id ?? "";
        setSelectedAnswerModel(defaultModel);
      })
      .catch((modelError) => {
        if (active) setError(modelError instanceof Error ? modelError.message : "Unable to load answer models");
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    listWorkingSets()
      .then((resp) => { if (active) setWorkingSets(resp.working_sets ?? []); })
      .catch(() => { if (active) setWorkingSets([]); });
    return () => { active = false; };
  }, []);

  function resolveSourceIds(): string[] | undefined {
    if (!selectedWorkingSetId) return undefined;
    const ws = workingSets.find((w) => w.id === selectedWorkingSetId);
    return ws?.source_ids?.length ? ws.source_ids : undefined;
  }

  function resolveParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    const sc = parseInt(seedCount, 10);
    const rc = parseInt(resultCount, 10);
    const rk = parseInt(rrfK, 10);
    if (!isNaN(sc)) params.seed_count = sc;
    if (!isNaN(rc)) params.result_count = rc;
    if (!isNaN(rk)) params.rrf_k = rk;
    const sourceIds = resolveSourceIds();
    if (sourceIds?.length) params.source_ids = sourceIds;
    return params;
  }

  function buildEvidence(): EvidenceItem[] {
    return retrieveResults.map((r) => ({
      source_id: r.source_id,
      source_name:
        r.source_metadata.source ||
        r.source_metadata.title ||
        r.source_metadata.author ||
        "Unknown source",
      text: r.chunk,
    }));
  }

  async function handleSubmit() {
    if (!query.trim()) return;
    setIsLoading(true);
    setError(null);
    setAnswerText("");
    setRetrieveResults([]);
    setSavedId(null);
    try {
      const sourceIds = resolveSourceIds();
      const sc = parseInt(seedCount, 10);
      const rc = parseInt(resultCount, 10);
      const rk = parseInt(rrfK, 10);
      await streamAnswer({
        query: query.trim(),
        model: selectedAnswerModel,
        ...(sourceIds?.length ? { source_ids: sourceIds } : {}),
        ...(!isNaN(sc) ? { seed_count: sc } : {}),
        ...(!isNaN(rc) ? { result_count: rc } : {}),
        ...(!isNaN(rk) ? { rrf_k: rk } : {}),
        onAnswerDelta: (delta) => setAnswerText((current) => current + delta),
        onResults: (results) => setRetrieveResults(results),
      });
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
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
    setAnswerText("");
    setRetrieveResults([]);
    setError(null);
    setSavedId(null);
  }

  async function handleSave() {
    if (!answerText || !query.trim()) return;
    setSaving(true);
    try {
      const resp = await saveAnswer(
        query.trim(),
        answerText,
        selectedAnswerModel,
        resolveParams(),
        buildEvidence(),
      );
      setSavedId(resp.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save answer");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="content">
      <form className="query-panel" onSubmit={handleFormSubmit} aria-label="Answer form">
        <label className="query-panel__label" htmlFor="semantic-query">
          Semantic Query
        </label>
        <div className="search-controls">
          <label className="search-controls__field" htmlFor="answer-model">
            <span>Answer Model</span>
            <select
              aria-label="Answer Model"
              id="answer-model"
              value={selectedAnswerModel}
              onChange={(e) => setSelectedAnswerModel(e.target.value)}
            >
              {answerModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>
          <label className="search-controls__field" htmlFor="answer-working-set">
            <span>Working Set (optional)</span>
            <select
              id="answer-working-set"
              value={selectedWorkingSetId}
              onChange={(e) => setSelectedWorkingSetId(e.target.value)}
            >
              <option value="">All sources</option>
              {workingSets.map((ws) => (
                <option key={ws.id} value={ws.id}>
                  {ws.name} ({ws.source_count} sources)
                </option>
              ))}
            </select>
          </label>
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
          <div className="search-controls community-retrieve-controls">
            <label className="search-controls__field" htmlFor="answer-seed-count">
              <span>Seed Count</span>
              <input
                id="answer-seed-count"
                min="1"
                step="1"
                type="number"
                value={seedCount}
                onChange={(e) => setSeedCount(e.target.value)}
              />
              <span className="arg-help">Maximum number of seed chunks used for graph expansion.</span>
            </label>
            <label className="search-controls__field" htmlFor="answer-result-count">
              <span>Result Count</span>
              <input
                id="answer-result-count"
                min="1"
                step="1"
                type="number"
                value={resultCount}
                onChange={(e) => setResultCount(e.target.value)}
              />
              <span className="arg-help">Maximum number of final results to return.</span>
            </label>
            <label className="search-controls__field" htmlFor="answer-rrf-k">
              <span>RRF K</span>
              <input
                id="answer-rrf-k"
                min="1"
                step="1"
                type="number"
                value={rrfK}
                onChange={(e) => setRrfK(e.target.value)}
              />
              <span className="arg-help">Reciprocal Rank Fusion parameter; higher values give more weight to dense results.</span>
            </label>
          </div>
        ) : null}

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
              Answer
            </button>
          </div>
        </div>
      </form>

      <section className="answer-panel">
        <div className="answer-panel__header">
          <h3>Answer</h3>
          {answerText && !isLoading ? (
            <div className="community-actions">
              <button
                className="primary-button"
                type="button"
                disabled={saving}
                onClick={handleSave}
              >
                {saving ? "Saving..." : "Save"}
              </button>
              {savedId ? (
                <span className="panel-state" role="status">
                  Answer saved! View in Library.
                </span>
              ) : null}
            </div>
          ) : null}
        </div>
        <div className="answer-panel__body">
          {answerText ? answerText : isLoading ? "Thinking..." : "Run Answer to generate a grounded response."}
        </div>
      </section>

      {retrieveResults.length > 0 ? (
        <section className="results-panel">
          <div className="results-panel__header">
            <h3>Evidence</h3>
            <span>{retrieveResults.length} result{retrieveResults.length === 1 ? "" : "s"}</span>
          </div>
          {error ? <p className="panel-state panel-state--error">{error}</p> : null}
          <div className="results-stack">
            {retrieveResults.map((result) => (
              <ResultCard key={result.chunk_id} result={result} onCopyChunk={onCopyChunk} onView={onView} />
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
