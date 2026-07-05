import { useCallback, useState } from "react";

import { useAuth } from "./auth/AuthContext";
import { Login } from "./auth/Login";
import SourcePanel from "./components/SourcePanel";
import type { SourceDetail } from "./lib/api";
import { getSource } from "./lib/api";
import SearchView from "./views/SearchView";
import RetrieveView from "./views/RetrieveView";
import AnswerView from "./views/AnswerView";
import CommunityView from "./views/CommunityView";
import ExploreView from "./views/ExploreView";
import LibraryView from "./views/LibraryView";
import WorkingSetsView from "./views/WorkingSetsView";

type Mode = "explore" | "search" | "retrieve" | "answer" | "community" | "library" | "working_sets";

type NavItem = {
  mode: Mode;
  label: string;
};

const NAV_ITEMS: NavItem[] = [
  { mode: "explore", label: "Explore" },
  { mode: "search", label: "Search" },
  { mode: "retrieve", label: "Retrieve" },
  { mode: "community", label: "Communities" },
  { mode: "answer", label: "Answer" },
  { mode: "library", label: "Library" },
  { mode: "working_sets", label: "Working Sets" },
];

export default function App() {
  const { state } = useAuth();
  if (state.status === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-300 text-sm">
        Loading…
      </div>
    );
  }
  if (state.status === "anonymous") {
    return <Login />;
  }
  return <Authenticated />;
}

function Authenticated() {
  const { logout, state } = useAuth();
  const username = state.status === "authenticated" ? state.user.username : "";
  const [mode, setMode] = useState<Mode>("explore");

  const [previewSource, setPreviewSource] = useState<SourceDetail | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const [communityScopeIds, setCommunityScopeIds] = useState<string[] | undefined>(undefined);
  const [retrieveScopeIds, setRetrieveScopeIds] = useState<string[] | undefined>(undefined);

  const handleView = useCallback(async (sourceId: string) => {
    setPreviewLoading(true);
    setPreviewError(null);
    setPreviewSource(null);
    try {
      const response = await getSource(sourceId);
      setPreviewSource(response);
    } catch (e) {
      setPreviewError(e instanceof Error ? e.message : "Unable to load source");
    } finally {
      setPreviewLoading(false);
    }
  }, []);

  function closePreview() {
    setPreviewSource(null);
    setPreviewError(null);
    setPreviewLoading(false);
  }

  function handleAddToBucket(_sourceId: string, _title: string) {
    // Retained for backward compat with ResultCard/InsightCard.
    // Add-to-working-set flow is in ExploreView.
  }

  function handleCopyChunk(chunk: string) {
    return navigator.clipboard.writeText(chunk);
  }

  function navigateCommunity(sourceIds: string[]) {
    setCommunityScopeIds(sourceIds);
    setMode("community");
  }

  function navigateRetrieve(sourceIds: string[]) {
    setRetrieveScopeIds(sourceIds);
    setMode("retrieve");
  }

  function switchMode(newMode: Mode) {
    setCommunityScopeIds(undefined);
    setRetrieveScopeIds(undefined);
    setMode(newMode);
  }

  return (
    <div className="app-shell">
      <nav className="sidebar" role="navigation" aria-label="Main navigation">
        <div className="sidebar__brand">
          <h1 className="sidebar__title">estebanf&apos;s RAG</h1>
          <p className="sidebar__subtitle">Knowledge graph retrieval</p>
        </div>
        <ul className="sidebar__nav">
          {NAV_ITEMS.map((item) => (
            <li key={item.mode}>
              <button
                className={`sidebar__nav-item${mode === item.mode ? " sidebar__nav-item--active" : ""}`}
                type="button"
                onClick={() => switchMode(item.mode)}
                aria-current={mode === item.mode ? "page" : undefined}
              >
                {item.label}
              </button>
            </li>
          ))}
        </ul>
        <div className="sidebar__footer">
          <div className="sidebar__user">
            <span>{username}</span>
            <button
              className="sidebar__logout"
              type="button"
              onClick={() => { void logout(); }}
            >
              Sign out
            </button>
          </div>
        </div>
      </nav>

      <div className="main-shell">
        {mode === "explore" ? (
          <ExploreView
            onView={handleView}
            onNavigateCommunity={navigateCommunity}
            onNavigateRetrieve={navigateRetrieve}
          />
        ) : mode === "search" ? (
          <SearchView
            onView={handleView}
            onAddToBucket={handleAddToBucket}
            onCopyChunk={handleCopyChunk}
          />
        ) : mode === "retrieve" ? (
          <RetrieveView
            onView={handleView}
            onAddToBucket={handleAddToBucket}
            onCopyChunk={handleCopyChunk}
          />
        ) : mode === "answer" ? (
          <AnswerView
            onView={handleView}
            onCopyChunk={handleCopyChunk}
          />
        ) : mode === "community" ? (
          <CommunityView
            onView={handleView}
            initialSourceIds={communityScopeIds}
          />
        ) : mode === "library" ? (
          <LibraryView onView={handleView} />
        ) : mode === "working_sets" ? (
          <WorkingSetsView
            onNavigateCommunity={navigateCommunity}
            onNavigateRetrieve={navigateRetrieve}
          />
        ) : null}
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
