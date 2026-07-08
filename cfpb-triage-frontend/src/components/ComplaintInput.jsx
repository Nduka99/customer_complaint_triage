import { useState } from "react";

// Example complaints spanning different product categories AND confidence levels.
// The first 3 are clear-cut cases (high confidence → automated routing).
// The last 2 are deliberately ambiguous — they blur boundaries between
// confused class pairs identified in NB09/NB11 and should trigger the
// low-confidence escalation path (referred for human review).
const EXAMPLES = [
  {
    label: "Hidden bank fees",
    text: "I opened a checking account and was charged hidden fees that were not disclosed during signup. The bank said these were standard maintenance fees but they were never mentioned in any of the paperwork I signed.",
    ambiguous: false,
  },
  {
    label: "Credit report error",
    text: "My credit report shows a late payment that I never made. I have proof of on-time payment including bank statements and receipts. I have disputed this with the credit bureau twice but nothing has changed.",
    ambiguous: false,
  },
  {
    label: "Debt collector harassment",
    text: "A debt collector keeps calling me about a debt I already paid off and is threatening legal action. I have sent them proof of payment three times but they refuse to acknowledge it and continue to call daily.",
    ambiguous: false,
  },
  {
    label: "Debt plan gone wrong",
    // Deliberately ambiguous: mixes debt collection language with credit management
    // themes. NB09 found Debt Mgmt ↔ Debt Collection is the #1 confused pair (28.7%).
    text: "A company enrolled me in a debt management plan but then sold my accounts to a collector without notice. Now I am getting calls about debts I thought were being handled. I am not sure if this is a debt collection issue or a problem with the credit management service I signed up for.",
    ambiguous: true,
  },
  {
    label: "Held money transfer",
    // Ambiguous between Money Transfer and Payday/Personal Loan — two weak classes.
    // NB09: Money Xfer ↔ Bank Acct confusion at 18.9%, Payday at 23.8% escalation.
    text: "I used an app to send money to pay back a personal loan from a friend but the transfer was held and the company says I need to verify the source of funds. They will not release my money or return it and I am stuck with the debt and the fees from the app.",
    ambiguous: true,
  },
];

export default function ComplaintInput({ onSubmit, loading }) {
  const [text, setText] = useState("");

  // Character count for the textarea — gives users a sense of input size
  const charCount = text.length;

  function handleSubmit(e) {
    e.preventDefault();
    if (!text.trim() || loading) return;
    onSubmit(text.trim());
  }

  function handleExample(exampleText) {
    setText(exampleText);
    // Auto-submit the example so the user sees results immediately
    onSubmit(exampleText);
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
        <div className="relative">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Describe what happened — the product or service involved and what went wrong…"
            rows={7}
            className="w-full bg-gray-900/60 border border-gray-700/50 rounded-xl px-4 py-3
                       text-gray-100 placeholder-gray-500 text-sm leading-relaxed
                       focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500/50
                       transition-all duration-200 resize-none"
          />
          {/* Character count — subtle, bottom-right of textarea */}
          <span className="absolute bottom-3 right-3 text-xs text-gray-600">
            {charCount > 0 && `${charCount.toLocaleString()} characters`}
          </span>
        </div>

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
              {/* Spinning indicator */}
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
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
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2.5 font-medium">
          Try an example
        </p>
        <div className="grid grid-cols-2 gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              onClick={() => handleExample(ex.text)}
              disabled={loading}
              className={`text-left px-3 py-2.5 rounded-lg text-sm
                         border disabled:opacity-40 disabled:cursor-not-allowed
                         transition-all duration-150
                         ${ex.ambiguous
                           ? "bg-amber-900/20 border-amber-700/30 text-amber-300 hover:bg-amber-900/30 hover:border-amber-600/50"
                           : "bg-gray-800/50 border-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-800 hover:border-gray-600/50"
                         }`}
            >
              {ex.label}
              {ex.ambiguous && (
                <span className="block text-xs text-amber-500/70 mt-0.5">
                  Ambiguous case — may be referred for review
                </span>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
