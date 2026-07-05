import { useEffect, useState } from "react";

import type { WorkingSet } from "../lib/api";
import {
  deleteWorkingSet,
  listWorkingSets,
  updateWorkingSet,
} from "../lib/api";

type WorkingSetsViewProps = {
  onNavigateCommunity: (sourceIds: string[]) => void;
  onNavigateRetrieve: (sourceIds: string[]) => void;
};

export default function WorkingSetsView({ onNavigateCommunity, onNavigateRetrieve }: WorkingSetsViewProps) {
  const [workingSets, setWorkingSets] = useState<WorkingSet[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const resp = await listWorkingSets();
      setWorkingSets(resp.working_sets ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load working sets");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleDelete(id: string) {
    try {
      await deleteWorkingSet(id);
      if (selectedId === id) setSelectedId(null);
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to delete");
    }
  }

  function startRename(ws: WorkingSet) {
    setRenamingId(ws.id);
    setRenameValue(ws.name);
  }

  async function handleRename(id: string) {
    if (!renameValue.trim()) return;
    try {
      await updateWorkingSet(id, { name: renameValue.trim() });
      setRenamingId(null);
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to rename");
    }
  }

  async function handleRemoveSource(wsId: string, sourceId: string) {
    const ws = workingSets.find((w) => w.id === wsId);
    if (!ws) return;
    const newIds = ws.source_ids.filter((id) => id !== sourceId);
    try {
      await updateWorkingSet(wsId, { source_ids: newIds });
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to remove source");
    }
  }

  const selected = workingSets.find((ws) => ws.id === selectedId) || null;

  return (
    <div className="content">
      <div className="explore-shell">
        <aside className="explore-facets" aria-label="Working sets list">
          <div className="explore-facets__header">
            <p className="eyebrow">Working Sets</p>
          </div>
          {loading ? <p className="panel-state">Loading...</p> : null}
          {error ? <p className="panel-state panel-state--error">{error}</p> : null}
          {!loading && !error && workingSets.length === 0 ? (
            <p className="panel-state">No working sets yet — select sources in Explore to create one</p>
          ) : null}
          <div className="source-list">
            {workingSets.map((ws) => (
              <article
                className={`source-list__item${ws.id === selectedId ? " source-list__item--active" : ""}`}
                key={ws.id}
              >
                <button
                  className="source-list__select"
                  type="button"
                  onClick={() => setSelectedId(ws.id)}
                >
                  <span className="source-list__title">{ws.name}</span>
                  <span className="source-list__meta">
                    <span>{ws.source_count} source{ws.source_count === 1 ? "" : "s"}</span>
                    <span className="source-list__id">{ws.id.slice(0, 8)}</span>
                  </span>
                </button>
                <div className="source-list__attributes">
                  <button
                    className="source-list__attribute"
                    type="button"
                    onClick={() => startRename(ws)}
                  >
                    <strong>Rename</strong>
                  </button>
                  <button
                    className="source-list__attribute"
                    type="button"
                    onClick={() => handleDelete(ws.id)}
                  >
                    <strong>Delete</strong>
                  </button>
                </div>
              </article>
            ))}
          </div>
        </aside>

        <section className="explore-sources" aria-label="Working set detail">
          {selected ? (
            <>
              <div className="explore-sources__header">
                <div>
                  <p className="eyebrow">Working Set</p>
                  {renamingId === selected.id ? (
                    <div className="rename-row">
                      <input
                        className="source-search-input"
                        type="text"
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleRename(selected.id);
                          if (e.key === "Escape") setRenamingId(null);
                        }}
                      />
                      <button className="ghost-button" type="button" onClick={() => handleRename(selected.id)}>
                        Save
                      </button>
                      <button className="ghost-button" type="button" onClick={() => setRenamingId(null)}>
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <h2>{selected.name}</h2>
                  )}
                </div>
                <span>{selected.source_count} sources</span>
              </div>

              <div className="selection-bar">
                <button
                  className="ghost-button"
                  type="button"
                  onClick={() => onNavigateCommunity(selected.source_ids)}
                >
                  Run communities
                </button>
                <button
                  className="ghost-button"
                  type="button"
                  onClick={() => onNavigateRetrieve(selected.source_ids)}
                >
                  Run retrieve
                </button>
              </div>

              <div className="source-list">
                {selected.source_ids.map((sourceId, i) => (
                  <article className="source-list__item" key={sourceId}>
                    <div className="source-list__row">
                      <span className="source-list__title">Source {i + 1}</span>
                      <span className="source-list__meta">
                        <span className="source-list__id">{sourceId}</span>
                      </span>
                    </div>
                    <div className="source-list__attributes">
                      <button
                        className="source-list__attribute"
                        type="button"
                        onClick={() => handleRemoveSource(selected.id, sourceId)}
                      >
                        <strong>Remove</strong>
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </>
          ) : (
            <div className="explore-sources__header">
              <p className="panel-state">Select a working set to view details</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
