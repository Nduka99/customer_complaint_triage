import { useId, useState } from "react";

/**
 * RagContext — Displays the top regulation excerpts retrieved for this complaint.
 *
 * Each passage card shows:
 *   - Source document name (e.g. "CFPB EFTA Examination Procedures")
 *   - Issue category the passage relates to
 *   - Truncated passage text (first ~150 chars), expandable on click
 *
 * The backend returns rag_context as an array of objects:
 *   { text: string (up to 500 chars), source: string, issue: string }
 *
 * These are real CFPB regulatory passages, filtered by the predicted product
 * category and ranked by relevance. They ground the classification in actual
 * regulation — the system doesn't just label complaints, it surfaces the
 * legal context a human agent would need.
 */

export default function RagContext({ passages }) {
  if (!passages || passages.length === 0) return null;

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-100 tracking-tight mb-1">
        Relevant regulation
      </h2>
      <p className="text-xs text-gray-400 mb-3">
        Excerpts from CFPB examination procedures related to this complaint
      </p>
      <div className="space-y-2.5">
        {passages.map((passage, i) => (
          <PassageCard key={i} passage={passage} index={i} />
        ))}
      </div>
    </div>
  );
}

function PassageCard({ passage, index }) {
  const [expanded, setExpanded] = useState(false);
  const textId = useId();

  // Show first 150 chars when collapsed, full text when expanded
  const previewLength = 150;
  const isLong = passage.text.length > previewLength;
  const displayText = expanded ? passage.text : passage.text.slice(0, previewLength);

  // A plain card; only long excerpts get a real disclosure button.
  return (
    <div className="bg-gray-900/40 border border-gray-700 rounded-xl px-4 py-3">
      {/* Header row: source badge + rank */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-gray-200">
          <span aria-hidden="true" className="w-1.5 h-1.5 rounded-full bg-gray-400 shrink-0" />
          {passage.source}
        </span>
        <span className="text-xs text-gray-400 font-mono shrink-0">
          <span className="sr-only">Excerpt </span>#{index + 1}
        </span>
      </div>

      {/* Issue category */}
      <p className="text-xs text-gray-400 mb-1.5 italic">
        {passage.issue}
      </p>

      {/* Passage text — truncated with expand/collapse */}
      <p id={textId} className="text-sm text-gray-300 leading-relaxed max-w-prose">
        {displayText}
        {isLong && !expanded && "…"}
      </p>

      {/* Expand/collapse control with chevron */}
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
          aria-controls={textId}
          className="mt-1.5 inline-flex items-center gap-1 text-xs text-blue-300 hover:text-blue-200"
        >
          <svg
            aria-hidden="true"
            className={`w-3 h-3 motion-safe:transition-transform motion-safe:duration-200 ${expanded ? "rotate-180" : ""}`}
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M5.23 7.21a.75.75 0 011.06.02L10 11.17l3.71-3.94a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"
              clipRule="evenodd"
            />
          </svg>
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}
