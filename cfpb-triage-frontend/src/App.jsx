import { useState, useEffect } from "react";
import { connectToBackend, classifyComplaint } from "./api";
import { CACHED_EXAMPLE, CACHED_ESCALATION } from "./cachedExample";
import ComplaintInput from "./components/ComplaintInput";
import ResultsDashboard from "./components/ResultsDashboard";
import AgenticTrace from "./components/AgenticTrace";
import RagContext from "./components/RagContext";
import LoadingState from "./components/LoadingState";

/**
 * App — Root component that orchestrates the two-column dashboard layout.
 *
 * Layout (desktop):
 *   Left column:  ComplaintInput (textarea + example chips)
 *   Right column: ResultsDashboard → AgenticTrace → RagContext
 *
 * Layout (mobile):
 *   Single column stack: Input → Results → Trace → RAG
 *
 * State machine for backend connection:
 *   "connecting" → trying to reach the analysis service (may be waking from sleep)
 *   "ready"      → backend is awake, user can submit complaints
 *   "error"      → connection failed; user can retry via LoadingState
 *
 * Two-layer UX: all user-facing copy is plain product language. The raw
 * pipeline internals (model names, sampling values, thresholds) are only
 * revealed when the user flips the "Technical details" toggle, which is
 * passed down as `showTech` to the result components.
 */

function App() {
  // Backend connection state — drives the LoadingState component
  const [backendStatus, setBackendStatus] = useState("connecting");
  const [backendError, setBackendError] = useState(null);
  // Incremented by the Retry button — re-runs the connection effect
  const [retryToken, setRetryToken] = useState(0);

  // Classification state — drives the results panel
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Layer-2 toggle: reveals raw pipeline internals in the result components
  const [showTech, setShowTech] = useState(false);

  // On mount (and on retry), attempt to connect to the backend.
  // If the service is waking from sleep, this can take a couple of minutes.
  // Once connected, we flip status to "ready" and the LoadingState disappears.
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        await connectToBackend();
        if (!cancelled) setBackendStatus("ready");
      } catch (err) {
        if (!cancelled) {
          // Raw error goes to the console for debugging; the user sees
          // friendly copy inside LoadingState instead.
          console.error("Backend connection failed:", err);
          setBackendStatus("error");
        }
      }
    }

    init();
    return () => { cancelled = true; };
  }, [retryToken]);

  // Called by LoadingState's "Try again" button after a failed connection
  function handleRetry() {
    setBackendStatus("connecting");
    setBackendError(null);
    setRetryToken((t) => t + 1);
  }

  // Called by ComplaintInput when the user submits or clicks an example
  async function handleSubmit(text) {
    setLoading(true);
    setResult(null);
    setBackendError(null); // clear any stale error from a previous attempt
    try {
      const data = await classifyComplaint(text);
      setResult(data);
      // If this is the first successful request, ensure status is "ready"
      // (handles edge case where connection check timed out but request works)
      if (backendStatus !== "ready") setBackendStatus("ready");
    } catch (err) {
      // Keep the raw error out of the UI — log it, show friendly copy
      console.error("Classification request failed:", err);
      setBackendError(
        "Something went wrong while analyzing this complaint. Please try again."
      );
    }
    setLoading(false);
  }

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800/60 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">
              Complaint Triage
            </h1>
            <p className="text-xs text-gray-500 mt-0.5">
              Intelligent routing for consumer financial complaints
            </p>
          </div>
          {/* Backend status indicator */}
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span
              className={`w-2 h-2 rounded-full ${
                backendStatus === "ready"
                  ? "bg-emerald-500"
                  : backendStatus === "error"
                  ? "bg-red-500"
                  : "bg-yellow-500 animate-pulse"
              }`}
            />
            {backendStatus === "ready"
              ? "System online"
              : backendStatus === "error"
              ? "Offline"
              : "Starting up…"}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Show loading state if backend isn't ready yet.
            While waiting, display cached sample results so visitors can see
            what the system does before the backend finishes starting. */}
        {backendStatus !== "ready" && (
          <>
            <LoadingState status={backendStatus} onRetry={handleRetry} />
            {backendStatus === "connecting" && (
              <div className="mt-8">
                <div className="flex items-center gap-2 mb-6">
                  <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Sample results while the system starts
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-blue-500/15 text-blue-400 uppercase tracking-wider">
                    Sample
                  </span>
                </div>
                {/* Two cached results side by side: automated vs escalated */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 opacity-70">
                  {/* Left: high-confidence automated example */}
                  <div className="space-y-4">
                    <p className="text-xs text-emerald-500/70 font-medium uppercase tracking-wider">
                      High confidence — handled automatically
                    </p>
                    <ResultsDashboard result={CACHED_EXAMPLE} />
                    <AgenticTrace trace={CACHED_EXAMPLE.agentic_trace} />
                  </div>
                  {/* Right: low-confidence escalation example */}
                  <div className="space-y-4">
                    <p className="text-xs text-amber-500/70 font-medium uppercase tracking-wider">
                      Low confidence — referred for review
                    </p>
                    <ResultsDashboard result={CACHED_ESCALATION} />
                    <AgenticTrace trace={CACHED_ESCALATION.agentic_trace} />
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* Two-column layout: input (left) + results (right) */}
        {backendStatus === "ready" && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
            {/* Left column — 2/5 width on desktop */}
            <div className="lg:col-span-2">
              <ComplaintInput onSubmit={handleSubmit} loading={loading} />
            </div>

            {/* Right column — 3/5 width on desktop */}
            <div className="lg:col-span-3 space-y-6">
              {/* Friendly error message for a failed classification */}
              {backendError && !result && (
                <p className="text-red-400 text-sm">{backendError}</p>
              )}

              {/* Empty state — before any classification */}
              {!result && !loading && (
                <div className="flex items-center justify-center h-64 border border-dashed border-gray-800 rounded-xl">
                  <p className="text-gray-600 text-sm">
                    Submit a complaint or try an example to see how it&rsquo;s routed
                  </p>
                </div>
              )}

              {/* Loading state for individual requests */}
              {loading && (
                <div className="flex items-center justify-center h-64">
                  <div className="flex items-center gap-3 text-gray-400">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    <span className="text-sm">Analyzing complaint…</span>
                  </div>
                </div>
              )}

              {/* Results — only shown after a successful classification */}
              {result && !loading && (
                <div className="space-y-6">
                  {/* Layer-2 toggle — reveals model names, sampling values, thresholds */}
                  <div className="flex justify-end">
                    <button
                      onClick={() => setShowTech((s) => !s)}
                      className="text-xs px-3 py-1 rounded-full border transition-colors duration-150
                        border-gray-700/60 text-gray-500 hover:text-gray-300 hover:border-gray-600"
                    >
                      {showTech ? "Hide technical details" : "Show technical details"}
                    </button>
                  </div>
                  <ResultsDashboard result={result} showTech={showTech} />
                  <AgenticTrace trace={result.agentic_trace} showTech={showTech} />
                  <RagContext passages={result.rag_context} />
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800/60 px-6 py-4 mt-auto">
        <div className="max-w-7xl mx-auto text-center text-xs text-gray-600">
          Built on the CFPB Consumer Complaint Database
        </div>
      </footer>
    </div>
  );
}

export default App;
