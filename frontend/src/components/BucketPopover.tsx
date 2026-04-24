import { useEffect, useRef, useState } from "react";

type BucketEntry = { sourceId: string; title: string };

type BucketPopoverProps = {
  entries: BucketEntry[];
  onClear: () => void;
  onCopy: () => Promise<void>;
};

export default function BucketPopover({ entries, onClear, onCopy }: BucketPopoverProps) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
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

  async function handleCopy() {
    await onCopy();
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="bucket-anchor" ref={containerRef}>
      <button
        aria-label="Source bucket"
        className="icon-button"
        type="button"
        onClick={() => setOpen((prev) => !prev)}
      >
        ⊟
        {entries.length > 0 ? (
          <span className="bucket-count">{entries.length}</span>
        ) : null}
      </button>
      {open ? (
        <div aria-label="Collected sources" className="bucket-popover" role="dialog">
          <p className="bucket-popover__label">
            {entries.length === 0
              ? "No sources collected"
              : `${entries.length} source${entries.length === 1 ? "" : "s"}`}
          </p>
          {entries.length > 0 ? (
            <ul className="bucket-popover__list">
              {entries.map((e) => (
                <li className="bucket-popover__item" data-testid="bucket-entry" key={e.sourceId}>
                  <span className="bucket-popover__title">{e.title}</span>
                  <span className="bucket-popover__id">{e.sourceId}</span>
                </li>
              ))}
            </ul>
          ) : null}
          <div className="bucket-popover__actions">
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                onClear();
                setOpen(false);
              }}
            >
              Clear
            </button>
            <div className="feedback-anchor">
              <button
                className="primary-button"
                disabled={entries.length === 0}
                type="button"
                onClick={handleCopy}
              >
                Copy
              </button>
              {copied ? (
                <span className="copy-popper" role="status">
                  IDs copied
                </span>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
