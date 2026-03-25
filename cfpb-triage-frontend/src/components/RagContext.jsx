import { useState } from "react";

/**
 * RagContext — Displays the top-3 regulatory passages retrieved by BM25 (NB10).
 *
 * Each passage card shows:
 *   - Source document name (e.g. "CFPB EFTA Examination Procedures")
 *   - Issue category the passage relates to
 *   - Truncated passage text (first ~150 chars), expandable on click
 *
 * The backend returns rag_context as an array of objects:
 *   { text: string (up to 500 chars), source: string, issue: string }
 *
 * These are real CFPB regulatory passages from the knowledge base built
 * in NB10, filtered by predicted product category and ranked by BM25.
 * They ground the classification in actual regulation — showing the
 * employer that the system doesn't just label complaints, it retrieves
 * the relevant legal context a human agent would need.
 */

export default function RagContext({ passages }) {
  if (!passages || passages.length === 0) return null;

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-100 tracking-tight mb-1">
        Regulatory Context
      </h2>
      <p className="text-xs text-gray-500 mb-3">
        Top passages retrieved from CFPB regulatory documents via BM25
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

  // Show first 150 chars when collapsed, full text when expanded
  const previewLength = 150;
  const isLong = passage.text.length > previewLength;
  const displayText = expanded ? passage.text : passage.text.slice(0, previewLength);

  return (
    <button
      onClick={() => isLong && setExpanded(!expanded)}
      className={`w-full text-left bg-gray-900/40 border border-gray-700/40 rounded-xl
                  px-4 py-3 transition-all duration-200
                  ${isLong ? "cursor-pointer hover:border-gray-600/60" : "cursor-default"}`}
    >
      {/* Header row: source badge + issue */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-teal-400">
          <span className="w-1.5 h-1.5 rounded-full bg-teal-400 shrink-0" />
          {passage.source}
        </span>
        <span className="text-[10px] text-gray-500 font-mono shrink-0">
          #{index + 1}
        </span>
      </div>

      {/* Issue category */}
      <p className="text-xs text-gray-500 mb-1.5 italic">
        {passage.issue}
      </p>

      {/* Passage text — truncated with expand/collapse */}
      <p className="text-sm text-gray-400 leading-relaxed">
        {displayText}
        {isLong && !expanded && (
          <span className="text-teal-500 ml-1">...click to expand</span>
        )}
        {isLong && expanded && (
          <span className="text-teal-500 ml-1">...click to collapse</span>
        )}
      </p>
    </button>
  );
}
