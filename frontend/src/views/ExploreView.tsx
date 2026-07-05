import { useEffect, useState } from "react";

import type {
  FacetsResponse,
  FacetValue,
  MetadataFilter,
  SourceSummary,
  WorkingSet,
} from "../lib/api";
import {
  createWorkingSet,
  getFacets,
  listSources,
  listWorkingSets,
} from "../lib/api";

type ExploreViewProps = {
  onView: (sourceId: string) => void;
  onNavigateCommunity: (sourceIds: string[]) => void;
  onNavigateRetrieve: (sourceIds: string[]) => void;
};

const SOURCES_PAGE_SIZE = 20;
const FACET_KEYS = ["kind", "author", "source", "domain"];

export default function ExploreView({ onView, onNavigateCommunity, onNavigateRetrieve }: ExploreViewProps) {
  const [facets, setFacets] = useState<FacetsResponse["facets"] | null>(null);
  const [facetsLoading, setFacetsLoading] = useState(true);
  const [selectedFacetValues, setSelectedFacetValues] = useState<Record<string, string[]>>({});

  const [sources, setSources] = useState<SourceSummary[]>([]);
  const [sourcesTotal, setSourcesTotal] = useState(0);
  const [sourcesOffset, setSourcesOffset] = useState(0);
  const [sourcesLoading, setSourcesLoading] = useState(false);
  const [sourcesError, setSourcesError] = useState<string | null>(null);

  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);

  const [showWorkingSetPicker, setShowWorkingSetPicker] = useState(false);
  const [workingSets, setWorkingSets] = useState<WorkingSet[]>([]);
  const [newWorkingSetName, setNewWorkingSetName] = useState("");
  const [workingSetPickerMode, setWorkingSetPickerMode] = useState<"create" | "add">("create");
  const [selectedWorkingSetId, setSelectedWorkingSetId] = useState("");

  useEffect(() => {
    let active = true;
    getFacets()
      .then((resp) => { if (active) setFacets(resp.facets ?? {}); })
      .catch(() => {})
      .finally(() => { if (active) setFacetsLoading(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    setSourcesLoading(true);
    setSourcesError(null);

    const filters: MetadataFilter[] = [];
    for (const key of FACET_KEYS) {
      const values = selectedFacetValues[key] || [];
      for (const value of values) {
        filters.push({ key, value });
      }
    }

    listSources(SOURCES_PAGE_SIZE, sourcesOffset, filters)
      .then((resp) => {
        if (!active) return;
        setSources(resp.sources ?? []);
        setSourcesTotal(resp.total ?? 0);
      })
      .catch((e) => {
        if (active) setSourcesError(e instanceof Error ? e.message : "Failed to load sources");
      })
      .finally(() => {
        if (active) setSourcesLoading(false);
      });
    return () => { active = false; };
  }, [sourcesOffset, selectedFacetValues]);

  function toggleFacetValue(key: string, value: string) {
    setSourcesOffset(0);
    setSelectedFacetValues((prev) => {
      const current = prev[key] || [];
      if (current.includes(value)) {
        return { ...prev, [key]: current.filter((v) => v !== value) };
      }
      return { ...prev, [key]: [...current, value] };
    });
  }

  function toggleSourceSelection(sourceId: string) {
    setSelectedSourceIds((prev) =>
      prev.includes(sourceId) ? prev.filter((id) => id !== sourceId) : [...prev, sourceId],
    );
  }

  function handleSelectAll() {
    const currentIds = new Set(selectedSourceIds);
    let allOnPageSelected = true;
    for (const s of sources) {
      if (!currentIds.has(s.source_id)) {
        allOnPageSelected = false;
        break;
      }
    }
    if (allOnPageSelected) {
      setSelectedSourceIds((prev) =>
        prev.filter((id) => !sources.some((s) => s.source_id === id)),
      );
    } else {
      for (const s of sources) {
        currentIds.add(s.source_id);
      }
      setSelectedSourceIds(Array.from(currentIds));
    }
  }

  async function loadWorkingSets() {
    try {
      const resp = await listWorkingSets();
      setWorkingSets(resp.working_sets);
    } catch {
      setWorkingSets([]);
    }
  }

  function openWorkingSetPicker(mode: "create" | "add") {
    setWorkingSetPickerMode(mode);
    setNewWorkingSetName("");
    setSelectedWorkingSetId("");
    setShowWorkingSetPicker(true);
    loadWorkingSets();
  }

  async function handleCreateWorkingSet() {
    if (!newWorkingSetName.trim() || selectedSourceIds.length === 0) return;
    try {
      await createWorkingSet(newWorkingSetName.trim(), selectedSourceIds);
      setShowWorkingSetPicker(false);
      setSelectedSourceIds([]);
      setNewWorkingSetName("");
    } catch {
      // error silently
    }
  }

  function formatDate(value: string): string {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString(undefined, {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  }

  function sourceOrigin(s: SourceSummary): string {
    return s.metadata?.source || s.metadata?.author || s.file_name || "Unknown";
  }

  function sourceKind(s: SourceSummary): string {
    return s.metadata?.kind || "unknown";
  }

  const page = Math.floor(sourcesOffset / SOURCES_PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil(sourcesTotal / SOURCES_PAGE_SIZE));
  const allOnPageSelected = sources.length > 0 && sources.every((s) => selectedSourceIds.includes(s.source_id));

  return (
    <div className="content">
      <div className="explore-shell">
        <aside className="explore-facets" aria-label="Facet filters">
          <div className="explore-facets__header">
            <p className="eyebrow">Filters</p>
            <span className="facet-caption">counts are corpus-wide</span>
          </div>
          {facetsLoading ? (
            <p className="panel-state">Loading facets...</p>
          ) : facets ? (
            <div className="facet-groups">
              {FACET_KEYS.map((key) => {
                const values = facets[key];
                if (!values || values.length === 0) return null;
                const selected = selectedFacetValues[key] || [];
                return (
                  <div className="facet-group" key={key}>
                    <h4 className="facet-group__label">{key}</h4>
                    <ul className="facet-group__list">
                      {values.map((fv: FacetValue) => (
                        <li className="facet-group__item" key={`${key}-${fv.value}`}>
                          <label className="facet-checkbox">
                            <input
                              type="checkbox"
                              checked={selected.includes(fv.value)}
                              onChange={() => toggleFacetValue(key, fv.value)}
                            />
                            <span className="facet-checkbox__label">
                              {fv.value || "(none)"}
                            </span>
                            <span className="facet-checkbox__count">{fv.count}</span>
                          </label>
                        </li>
                      ))}
                    </ul>
                  </div>
                );
              })}
            </div>
          ) : null}
        </aside>

        <section className="explore-sources" aria-label="Source list">
          <div className="explore-sources__header">
            <div>
              <p className="eyebrow">Sources</p>
              <h2>
                {Object.values(selectedFacetValues).some((arr) => arr.length > 0)
                  ? "Filtered"
                  : "Latest"}
              </h2>
            </div>
            <span>{sourcesTotal} results</span>
          </div>

          {selectedSourceIds.length > 0 ? (
            <div className="selection-bar" aria-label="Selection actions">
              <span>{selectedSourceIds.length} selected</span>
              <button
                className="ghost-button"
                type="button"
                onClick={() => openWorkingSetPicker("create")}
              >
                + New working set
              </button>
              <button
                className="ghost-button"
                type="button"
                onClick={() => openWorkingSetPicker("add")}
              >
                Add to existing
              </button>
              <button
                className="ghost-button"
                type="button"
                onClick={() => onNavigateCommunity(selectedSourceIds)}
              >
                Run communities
              </button>
              <button
                className="secondary-button"
                type="button"
                onClick={() => setSelectedSourceIds([])}
              >
                Clear
              </button>
            </div>
          ) : null}

          {showWorkingSetPicker ? (
            <div className="working-set-picker" role="dialog" aria-label="Working set picker">
              <div className="working-set-picker__header">
                <h4>
                  {workingSetPickerMode === "create"
                    ? "Create working set"
                    : "Add to working set"}
                </h4>
                <button
                  className="close-button"
                  type="button"
                  onClick={() => setShowWorkingSetPicker(false)}
                >
                  Close
                </button>
              </div>
              {workingSetPickerMode === "create" ? (
                <div className="working-set-picker__body">
                  <label className="search-controls__field" htmlFor="ws-name">
                    <span>Name</span>
                    <input
                      id="ws-name"
                      type="text"
                      value={newWorkingSetName}
                      onChange={(e) => setNewWorkingSetName(e.target.value)}
                      placeholder="My working set"
                    />
                  </label>
                  <button
                    className="primary-button"
                    type="button"
                    disabled={!newWorkingSetName.trim()}
                    onClick={handleCreateWorkingSet}
                  >
                    Create
                  </button>
                </div>
              ) : (
                <div className="working-set-picker__body">
                  {workingSets.length === 0 ? (
                    <p className="panel-state">No working sets yet.</p>
                  ) : (
                    <ul className="working-set-picker__list">
                      {workingSets.map((ws) => (
                        <li key={ws.id}>
                          <button
                            className="source-list__item"
                            type="button"
                            onClick={() => setSelectedWorkingSetId(ws.id)}
                          >
                            {ws.name} ({ws.source_count} sources)
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>
          ) : null}

          {sourcesLoading ? <p className="panel-state">Loading sources...</p> : null}
          {sourcesError ? <p className="panel-state panel-state--error">{sourcesError}</p> : null}

          {!sourcesLoading && !sourcesError && sources.length === 0 ? (
            <p className="panel-state">No sources match these facets — clear filters</p>
          ) : null}

          <div className="source-list">
            {sources.map((item) => (
              <article
                className={`source-list__item${selectedSourceIds.includes(item.source_id) ? " source-list__item--active" : ""}`}
                key={item.source_id}
              >
                <div className="source-list__row">
                  <label className="facet-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedSourceIds.includes(item.source_id)}
                      onChange={() => toggleSourceSelection(item.source_id)}
                    />
                  </label>
                  <button
                    className="source-list__select"
                    type="button"
                    onClick={() => onView(item.source_id)}
                  >
                    <span className="source-list__title">
                      {item.name || item.file_name || item.source_id}
                    </span>
                    <span className="source-list__meta">
                      <span>{sourceOrigin(item)} · {sourceKind(item)} · {formatDate(item.created_at)}</span>
                      <span className="source-list__id">{item.source_id}</span>
                    </span>
                  </button>
                </div>
              </article>
            ))}
          </div>

          <div className="source-paginator" aria-label="Sources pagination">
            <button
              className="source-paginator__button"
              disabled={page <= 1 || sourcesLoading}
              type="button"
              onClick={() => setSourcesOffset((prev) => Math.max(0, prev - SOURCES_PAGE_SIZE))}
            >
              Previous
            </button>
            <span>
              <label className="facet-checkbox source-select-all">
                <input
                  type="checkbox"
                  checked={allOnPageSelected}
                  onChange={handleSelectAll}
                />
                <span>Page {page} of {pageCount}</span>
              </label>
            </span>
            <button
              className="source-paginator__button"
              disabled={page >= pageCount || sourcesLoading}
              type="button"
              onClick={() => setSourcesOffset((prev) => prev + SOURCES_PAGE_SIZE)}
            >
              Next
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
