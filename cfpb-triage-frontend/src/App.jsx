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
 *   "connecting" → trying to reach HF Space (may be sleeping)
 *   "ready"      → backend is awake, user can submit complaints
 *   "error"      → connection failed after timeout
 */

function App() {
  // Backend connection state — drives the LoadingState component
  const [backendStatus, setBackendStatus] = useState("connecting");
  const [backendError, setBackendError] = useState(null);

  // Classification state — drives the results panel
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // On mount, attempt to connect to the HF Space backend.
  // If the Space is sleeping, this will hang until it wakes (~2 min).
  // Once connected, we flip status to "ready" and the LoadingState disappears.
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        await connectToBackend();
        if (!cancelled) setBackendStatus("ready");
      } catch (err) {
        if (!cancelled) {
          setBackendStatus("error");
          setBackendError(err.message);
        }
      }
    }

    init();
    return () => { cancelled = true; };
  }, []);

  // Called by ComplaintInput when the user submits or clicks an example
  async function handleSubmit(text) {
    setLoading(true);
    setResult(null);
    try {
      const data = await classifyComplaint(text);
      setResult(data);
      // If this is the first successful request, ensure status is "ready"
      // (handles edge case where connection check timed out but request works)
      if (backendStatus !== "ready") setBackendStatus("ready");
    } catch (err) {
      setBackendError(err.message);
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
              CFPB Agentic Triage System
            </h1>
            <p className="text-xs text-gray-500 mt-0.5">
              Ensemble classification + RL routing + RAG retrieval
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
              ? "Pipeline online"
              : backendStatus === "error"
              ? "Offline"
              : "Connecting..."}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Show loading state if backend isn't ready yet.
            While waiting, display the cached example result so employers
            can see what the system does before the backend finishes booting. */}
        {backendStatus !== "ready" && (
          <>
            <LoadingState status={backendStatus} error={backendError} />
            {backendStatus === "connecting" && (
              <div className="mt-8">
                <div className="flex items-center gap-2 mb-6">
                  <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Preview — cached pipeline outputs
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-blue-500/15 text-blue-400 uppercase tracking-wider">
                    Cached
                  </span>
                </div>
                {/* Two cached results side by side: automated vs escalated */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 opacity-70">
                  {/* Left: high-confidence automated example */}
                  <div className="space-y-4">
                    <p className="text-xs text-emerald-500/70 font-medium uppercase tracking-wider">
                      High confidence — automated
                    </p>
                    <ResultsDashboard result={CACHED_EXAMPLE} />
                    <AgenticTrace trace={CACHED_EXAMPLE.agentic_trace} />
                  </div>
                  {/* Right: low-confidence escalation example */}
                  <div className="space-y-4">
                    <p className="text-xs text-amber-500/70 font-medium uppercase tracking-wider">
                      Low confidence — escalated
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
              {/* Error message */}
              {backendError && !result && (
                <p className="text-red-400 text-sm">{backendError}</p>
              )}

              {/* Empty state — before any classification */}
              {!result && !loading && (
                <div className="flex items-center justify-center h-64 border border-dashed border-gray-800 rounded-xl">
                  <p className="text-gray-600 text-sm">
                    Submit a complaint or click an example to see results
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
                    <span className="text-sm">Running pipeline...</span>
                  </div>
                </div>
              )}

              {/* Results — only shown after a successful classification */}
              {result && !loading && (
                <div className="space-y-6">
                  <ResultsDashboard result={result} />
                  <AgenticTrace trace={result.agentic_trace} />
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
          BART-distilled RoBERTa + ModernBERT ensemble | Thompson Sampling routing | BM25 RAG
          <span className="mx-2">|</span>
          Built on the CFPB Consumer Complaint Database
        </div>
      </footer>
    </div>
  );
}

export default App;
