import { useEffect, useRef, useState } from "react";

import type { WorkingSet } from "../lib/api";
import { createWorkingSet, listWorkingSets, updateWorkingSet } from "../lib/api";

type AddToWorkingSetControlProps = {
  sourceId: string;
};

export default function AddToWorkingSetControl({ sourceId }: AddToWorkingSetControlProps) {
  const [open, setOpen] = useState(false);
  const [workingSets, setWorkingSets] = useState<WorkingSet[]>([]);
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [feedback, setFeedback] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleOutsideClick(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [open]);

  function toggleOpen() {
    const next = !open;
    setOpen(next);
    if (next) {
      setLoading(true);
      listWorkingSets()
        .then((resp) => setWorkingSets(resp.working_sets ?? []))
        .catch(() => setWorkingSets([]))
        .finally(() => setLoading(false));
    }
  }

  function flashFeedback(message: string) {
    setFeedback(message);
    window.setTimeout(() => setFeedback(null), 1800);
  }

  async function addToExisting(ws: WorkingSet) {
    if (ws.source_ids.includes(sourceId)) {
      flashFeedback(`Already in "${ws.name}"`);
      return;
    }
    setBusyId(ws.id);
    try {
      await updateWorkingSet(ws.id, { source_ids: [...ws.source_ids, sourceId] });
      flashFeedback(`Added to "${ws.name}"`);
      setOpen(false);
    } catch (e) {
      flashFeedback(e instanceof Error ? e.message : "Failed to add");
    } finally {
      setBusyId(null);
    }
  }

  async function createAndAdd() {
    const name = newName.trim();
    if (!name) return;
    setBusyId("__new__");
    try {
      await createWorkingSet(name, [sourceId]);
      flashFeedback(`Created "${name}" with this source`);
      setNewName("");
      setOpen(false);
    } catch (e) {
      flashFeedback(e instanceof Error ? e.message : "Failed to create working set");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="bucket-anchor" ref={containerRef}>
      <button
        aria-expanded={open}
        aria-label="Add to working set"
        className="ghost-button"
        type="button"
        onClick={toggleOpen}
      >
        + Working set
      </button>
      {open ? (
        <div aria-label="Add to working set" className="bucket-popover" role="region">
          <p className="bucket-popover__label">Add to working set</p>
          {loading ? <p className="panel-state">Loading...</p> : null}
          {!loading && workingSets.length === 0 ? (
            <p className="panel-state">No working sets yet — create one below.</p>
          ) : null}
          {!loading && workingSets.length > 0 ? (
            <ul className="bucket-popover__list">
              {workingSets.map((ws) => (
                <li className="bucket-popover__item" key={ws.id}>
                  <button
                    className="ghost-button working-set-picker__item"
                    disabled={busyId === ws.id}
                    type="button"
                    onClick={() => addToExisting(ws)}
                  >
                    <span className="bucket-popover__title">{ws.name}</span>
                    <span className="bucket-popover__id">{ws.source_count} sources</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
          <div className="working-set-picker__create">
            <input
              placeholder="New working set name"
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void createAndAdd();
                }
              }}
            />
            <button
              className="primary-button"
              disabled={!newName.trim() || busyId === "__new__"}
              type="button"
              onClick={createAndAdd}
            >
              Create &amp; add
            </button>
          </div>
          {feedback ? <p className="panel-state">{feedback}</p> : null}
        </div>
      ) : null}
    </div>
  );
}
