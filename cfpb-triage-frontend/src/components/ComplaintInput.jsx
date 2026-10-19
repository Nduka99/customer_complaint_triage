import { useState } from "react";
// Each example's label comes from what the live model was measured to do with it
// (see src/examples.js and scripts/check-examples.mjs).
import { EXAMPLES } from "../examples";

const EXAMPLE_TEXTS = new Set(EXAMPLES.map((example) => example.text));

export default function ComplaintInput({ onSubmit, loading }) {
  const [text, setText] = useState("");
  // The visitor's own text, kept when an example replaces it, so it can be restored.
  const [replacedText, setReplacedText] = useState(null);

  // Character count for the textarea — gives users a sense of input size
  const charCount = text.length;

  function handleSubmit(e) {
    e.preventDefault();
    if (!text.trim() || loading) return;
    onSubmit(text.trim());
  }

  function handleExample(exampleText) {
    // Keep anything the visitor typed themselves; an undo is kinder than a confirmation.
    if (text.trim() && !EXAMPLE_TEXTS.has(text)) setReplacedText(text);
    setText(exampleText);
    // Auto-submit the example so the user sees results immediately
    onSubmit(exampleText);
  }

  function restoreText() {
    setText(replacedText);
    setReplacedText(null);
  }

  return (
    <div className="space-y-5">
      {/* Section header */}
      <div>
        <h2 className="text-lg font-semibold text-gray-100 tracking-tight">
          Analyze a complaint
        </h2>
        <p className="text-sm text-gray-400 mt-1">
          Paste or type a consumer complaint to see how it&rsquo;s categorized
          and routed.
        </p>
      </div>

      {/* Textarea form */}
      <form onSubmit={handleSubmit}>
        <label htmlFor="complaint-text" className="block text-sm font-medium text-gray-200 mb-2">
          Complaint text
        </label>
        <div className="relative">
          <textarea
            id="complaint-text"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setReplacedText(null);
            }}
            aria-describedby="complaint-count"
            placeholder="Describe what happened: the product or service involved and what went wrong…"
            rows={7}
            className="w-full bg-gray-900/60 border border-gray-700 rounded-xl px-4 py-3
                       text-gray-100 placeholder-gray-400 text-sm leading-relaxed
                       focus:border-blue-400 resize-none"
          />
          {/* Character count — subtle, bottom-right of textarea */}
          <span id="complaint-count" className="absolute bottom-3 right-3 text-xs text-gray-400">
            {charCount > 0 && `${charCount.toLocaleString()} characters`}
          </span>
        </div>
        {replacedText !== null && (
          <button
            type="button"
            onClick={restoreText}
            className="mt-2 text-sm text-blue-300 underline underline-offset-2 hover:text-blue-200"
          >
            Restore your text
          </button>
        )}

        <button
          type="submit"
          disabled={!text.trim() || loading}
          className="mt-3 w-full py-2.5 rounded-lg text-sm font-medium
                     bg-blue-600 text-white
                     hover:bg-blue-500 active:bg-blue-700
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all duration-150
                     flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              {/* Spinning indicator (decorative: the button text says what is happening) */}
              <svg className="motion-safe:animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Analyzing…
            </>
          ) : (
            "Analyze complaint"
          )}
        </button>
      </form>

      {/* Example complaint chips */}
      <div role="group" aria-labelledby="examples-title">
        <p id="examples-title" className="text-xs text-gray-400 uppercase tracking-wider mb-2.5 font-medium">
          Try an example
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              type="button"
              onClick={() => handleExample(ex.text)}
              disabled={loading}
              className={`text-left px-3 py-2.5 rounded-lg text-sm
                         border disabled:opacity-40 disabled:cursor-not-allowed
                         transition-colors duration-150
                         ${ex.expect === "escalate"
                           ? "bg-amber-900/20 border-amber-700/50 text-amber-200 hover:bg-amber-900/40 hover:border-amber-500"
                           : "bg-gray-800/50 border-gray-700 text-gray-200 hover:text-white hover:bg-gray-800 hover:border-gray-500"
                         }`}
            >
              {ex.label}
              {ex.expect === "escalate" && (
                <span className="block text-xs text-amber-300 mt-0.5">
                  {ex.why}: goes to a person
                </span>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
