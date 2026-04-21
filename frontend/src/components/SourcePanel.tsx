import ReactMarkdown from "react-markdown";

import type { SourceDetail } from "../lib/api";

type SourcePanelProps = {
  source: SourceDetail | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
};

export default function SourcePanel({ source, loading, error, onClose }: SourcePanelProps) {
  return (
    <aside aria-label="Source Preview" className={`source-panel${source || loading || error ? " source-panel--open" : ""}`} role="complementary">
      <div className="source-panel__frame">
        <div className="source-panel__header">
          <div>
            <p className="eyebrow">Source preview</p>
            <h2>{source?.name || source?.file_name || "Markdown source"}</h2>
          </div>
          <button className="close-button" type="button" onClick={onClose}>
            Close
          </button>
        </div>
        {loading && <p className="panel-state">Loading source...</p>}
        {error && <p className="panel-state panel-state--error">{error}</p>}
        {source && !loading && !error ? (
          <div className="source-panel__body">
            <div className="source-panel__meta">
              <span>{source.metadata.source || "Local corpus"}</span>
              <span>{source.file_name}</span>
            </div>
            <ReactMarkdown>{source.markdown_content}</ReactMarkdown>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
