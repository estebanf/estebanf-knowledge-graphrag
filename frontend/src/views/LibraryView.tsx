import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

import type {
  EvidenceItem,
  SavedAnswer,
  ThemeReportDetail,
  ThemeReportSummary,
} from "../lib/api";
import {
  deleteAnswer,
  getAnswer,
  getTheme,
  listAnswers,
  listThemes,
  regenerateTheme,
} from "../lib/api";

type LibraryViewProps = {
  onView: (sourceId: string) => void;
};

type Tab = "themes" | "answers";

function confidenceStars(confidence: number): string {
  const n = Math.max(0, Math.min(5, Math.round(confidence)));
  return "★".repeat(n) + "☆".repeat(5 - n);
}

function statusBadgeClass(status: string): string {
  return `badge run-status-badge run-status-badge--${status}`;
}

export default function LibraryView({ onView }: LibraryViewProps) {
  const [tab, setTab] = useState<Tab>("themes");

  const [themes, setThemes] = useState<ThemeReportSummary[]>([]);
  const [themesLoading, setThemesLoading] = useState(true);
  const [themesError, setThemesError] = useState<string | null>(null);

  const [answers, setAnswers] = useState<SavedAnswer[]>([]);
  const [answersLoading, setAnswersLoading] = useState(true);
  const [answersError, setAnswersError] = useState<string | null>(null);

  const [selectedTheme, setSelectedTheme] = useState<ThemeReportDetail | null>(null);
  const [themeDetailLoading, setThemeDetailLoading] = useState(false);

  const [selectedAnswer, setSelectedAnswer] = useState<SavedAnswer | null>(null);
  const [answerDetailLoading, setAnswerDetailLoading] = useState(false);
  const [sourceStatuses, setSourceStatuses] = useState<Record<string, "available" | "unavailable">>({});

  const [regenerating, setRegenerating] = useState(false);

  useEffect(() => {
    let active = true;
    setThemesLoading(true);
    setThemesError(null);
    listThemes()
      .then((resp) => {
        if (active) setThemes(resp.reports || []);
      })
      .catch((e) => {
        if (active) setThemesError(e instanceof Error ? e.message : "Failed to load themes");
      })
      .finally(() => {
        if (active) setThemesLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    setAnswersLoading(true);
    setAnswersError(null);
    listAnswers()
      .then((resp) => {
        if (active) setAnswers(resp.answers || []);
      })
      .catch((e) => {
        if (active) setAnswersError(e instanceof Error ? e.message : "Failed to load answers");
      })
      .finally(() => {
        if (active) setAnswersLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  async function openTheme(id: string) {
    setThemeDetailLoading(true);
    setSelectedTheme(null);
    try {
      const report = await getTheme(id);
      setSelectedTheme(report);
    } catch (e) {
      setThemesError(e instanceof Error ? e.message : "Failed to load report");
    } finally {
      setThemeDetailLoading(false);
    }
  }

  async function openAnswer(id: string) {
    setAnswerDetailLoading(true);
    setSelectedAnswer(null);
    setSourceStatuses({});
    try {
      const ans = await getAnswer(id);
      setSelectedAnswer(ans);
      const statuses: Record<string, "available" | "unavailable"> = {};
      for (const item of ans.evidence_snapshot || []) {
        try {
          const resp = await fetch(`/api/sources/${item.source_id}`, {
            credentials: "include",
          });
          statuses[item.source_id] = resp.ok ? "available" : "unavailable";
        } catch {
          statuses[item.source_id] = "unavailable";
        }
      }
      setSourceStatuses(statuses);
    } catch (e) {
      setAnswersError(e instanceof Error ? e.message : "Failed to load answer");
    } finally {
      setAnswerDetailLoading(false);
    }
  }

  async function handleRegenerate(reportId: string) {
    setRegenerating(true);
    try {
      const resp = await regenerateTheme(reportId);
      setSelectedTheme(null);
      const report = await getTheme(resp.id);
      setSelectedTheme(report);
    } catch (e) {
      setThemesError(e instanceof Error ? e.message : "Failed to regenerate");
    } finally {
      setRegenerating(false);
    }
  }

  async function handleDeleteAnswer(id: string) {
    try {
      await deleteAnswer(id);
      setSelectedAnswer(null);
      setAnswers((prev) => prev.filter((a) => a.id !== id));
    } catch (e) {
      setAnswersError(e instanceof Error ? e.message : "Failed to delete");
    }
  }

  return (
    <div className="content">
      <div className="query-panel">
        <div className="search-controls">
          <button
            className={`ghost-button${tab === "themes" ? " sidebar__nav-item--active" : ""}`}
            type="button"
            onClick={() => setTab("themes")}
          >
            Theme Reports
          </button>
          <button
            className={`ghost-button${tab === "answers" ? " sidebar__nav-item--active" : ""}`}
            type="button"
            onClick={() => setTab("answers")}
          >
            Saved Answers
          </button>
        </div>
      </div>

      {tab === "themes" ? (
        <section className="results-panel">
          <div className="results-panel__header">
            <h3>Theme Reports</h3>
          </div>

          {selectedTheme ? (
            <div className="community-panel">
              <button className="ghost-button" type="button" onClick={() => setSelectedTheme(null)}>
                ← Back to reports
              </button>
              <div className="theme-report-detail">
                <div className="theme-report-detail__header">
                  <span className={statusBadgeClass(selectedTheme.status)}>
                    {selectedTheme.status}
                  </span>
                  <span className="run-history-item__id">{selectedTheme.run_id.slice(0, 8)}</span>
                  <span>{selectedTheme.model}</span>
                  <span className="run-history-item__date">
                    {new Date(selectedTheme.created_at).toLocaleString()}
                  </span>
                </div>

                {(selectedTheme.status === "partial" || selectedTheme.status === "failed") &&
                selectedTheme.failed_community_ids?.length ? (
                  <div className="theme-report-failed">
                    <p>
                      Failed community IDs: {selectedTheme.failed_community_ids.join(", ")}
                    </p>
                    <button
                      className="ghost-button"
                      type="button"
                      disabled={regenerating}
                      onClick={() => handleRegenerate(selectedTheme.id)}
                    >
                      {regenerating ? "Regenerating..." : "Regenerate"}
                    </button>
                  </div>
                ) : null}

                {selectedTheme.report.buckets?.length ? (
                  <div className="theme-buckets">
                    {selectedTheme.report.buckets.map((bucket, bi) => (
                      <div className="theme-bucket" key={`bucket-${bi}`}>
                        <h4 className="theme-bucket__label">{bucket.label}</h4>
                        <div className="community-cards">
                          {bucket.communities.map((c, ci) => (
                            <div className="community-card" key={`comm-${bi}-${ci}`}>
                              <div className="community-card__header">
                                <span className="badge">{c.label}</span>
                                {c.type ? (
                                  <span className="badge">{c.type}</span>
                                ) : null}
                                {c.cross_source ? (
                                  <span className="badge">Cross-source</span>
                                ) : null}
                              </div>
                              <div className="community-card__confidence">
                                {confidenceStars(c.confidence)}
                              </div>
                              {c.summary ? (
                                <p className="community-card__summary">{c.summary}</p>
                              ) : null}
                              {c.key_entities?.length ? (
                                <div className="community-card__entities">
                                  <strong>Key entities:</strong>{" "}
                                  {c.key_entities.join(", ")}
                                </div>
                              ) : null}
                              {c.key_sources?.length ? (
                                <div className="community-card__sources">
                                  <strong>Key sources:</strong>{" "}
                                  {c.key_sources.join(", ")}
                                </div>
                              ) : null}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : null}

                {selectedTheme.report.narrative ? (
                  <div className="theme-narrative">
                    <h4>Cross-community narrative</h4>
                    <ReactMarkdown>{selectedTheme.report.narrative}</ReactMarkdown>
                  </div>
                ) : null}

                {selectedTheme.report.cleanup_recommendations?.length ? (
                  <div className="theme-cleanup">
                    <h4>Cleanup recommendations</h4>
                    <ul>
                      {selectedTheme.report.cleanup_recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            </div>
          ) : themeDetailLoading ? (
            <p className="panel-state">Loading report...</p>
          ) : themesError ? (
            <p className="panel-state panel-state--error">{themesError}</p>
          ) : themesLoading ? (
            <p className="panel-state">Loading...</p>
          ) : themes.length === 0 ? (
            <p className="panel-state">
              No theme reports yet — generate one from a community run
            </p>
          ) : (
            <div className="run-history-list">
              {themes.map((theme) => (
                <div className="run-history-item" key={theme.id}>
                  <div className="run-history-item__info">
                    <span className={statusBadgeClass(theme.status)}>
                      {theme.status}
                    </span>
                    <span className="run-history-item__id">
                      {theme.run_id.slice(0, 8)}
                    </span>
                    <span>{theme.model}</span>
                    <span className="run-history-item__date">
                      {new Date(theme.created_at).toLocaleString()}
                    </span>
                  </div>
                  <button
                    className="ghost-button"
                    type="button"
                    onClick={() => openTheme(theme.id)}
                  >
                    View
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      ) : (
        <section className="results-panel">
          <div className="results-panel__header">
            <h3>Saved Answers</h3>
          </div>

          {selectedAnswer ? (
            <div className="community-panel">
              <button
                className="ghost-button"
                type="button"
                onClick={() => setSelectedAnswer(null)}
              >
                ← Back to answers
              </button>
              <div className="saved-answer-detail">
                <div className="theme-report-detail__header">
                  <span className="run-history-item__date">
                    {new Date(selectedAnswer.created_at).toLocaleString()}
                  </span>
                  <span>{selectedAnswer.model}</span>
                </div>
                <h4 className="saved-answer__question">{selectedAnswer.question}</h4>
                <div className="saved-answer__body">
                  <ReactMarkdown>{selectedAnswer.answer}</ReactMarkdown>
                </div>
                {selectedAnswer.evidence_snapshot?.length ? (
                  <div className="saved-answer__evidence">
                    <h4>Evidence</h4>
                    {selectedAnswer.evidence_snapshot.map((item, i) => {
                      const status =
                        sourceStatuses[item.source_id] || "available";
                      return (
                        <div className="saved-answer__evidence-item" key={`ev-${i}`}>
                          {status === "unavailable" ? (
                            <p className="panel-state panel-state--error">
                              source no longer available
                            </p>
                          ) : null}
                          <p className="saved-answer__evidence-text">{item.text}</p>
                          <div className="saved-answer__evidence-meta">
                            <span className="result-card__meta">
                              {item.source_name}
                            </span>
                            {status === "available" ? (
                              <button
                                className="ghost-button"
                                type="button"
                                onClick={() => onView(item.source_id)}
                              >
                                View
                              </button>
                            ) : null}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : null}
                <div className="community-actions">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => handleDeleteAnswer(selectedAnswer.id)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ) : answerDetailLoading ? (
            <p className="panel-state">Loading answer...</p>
          ) : answersError ? (
            <p className="panel-state panel-state--error">{answersError}</p>
          ) : answersLoading ? (
            <p className="panel-state">Loading...</p>
          ) : answers.length === 0 ? (
            <p className="panel-state">No saved answers yet</p>
          ) : (
            <div className="run-history-list">
              {answers.map((ans) => (
                <div className="run-history-item" key={ans.id}>
                  <div className="run-history-item__info">
                    <span className="run-history-item__question">
                      {ans.question.length > 80
                        ? `${ans.question.slice(0, 80)}...`
                        : ans.question}
                    </span>
                    <span>{ans.model}</span>
                    <span className="run-history-item__date">
                      {new Date(ans.created_at).toLocaleString()}
                    </span>
                  </div>
                  <button
                    className="ghost-button"
                    type="button"
                    onClick={() => openAnswer(ans.id)}
                  >
                    View
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
