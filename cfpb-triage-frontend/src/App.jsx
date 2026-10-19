import { useState, useEffect } from "react";
import { connectToBackend, classifyComplaint, getWakeStep } from "./api";
import {
  CACHED_EXAMPLE,
  CACHED_EXAMPLE_LABEL,
  CACHED_ESCALATION,
  CACHED_ESCALATION_LABEL,
} from "./cachedExample";
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

// A free Space sleeps when idle. The first request starts waking it, which can take a few
// minutes, so a failed connection is retried for as long as the Hub says the Space is waking.
const WAKE_LIMIT_MS = 5 * 60 * 1000;
const RETRY_DELAY_MS = 10 * 1000;

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
      const start = Date.now();
      while (!cancelled) {
        try {
          await connectToBackend();
          if (!cancelled) setBackendStatus("ready");
          return;
        } catch (err) {
          const step = await getWakeStep();
          const waking = step !== null && Date.now() - start < WAKE_LIMIT_MS;
          if (!waking) {
            // Raw error goes to the console for debugging; the user sees
            // friendly copy inside LoadingState instead.
            console.error("Backend connection failed:", err);
            if (!cancelled) setBackendStatus("error");
            return;
          }
          await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
        }
      }
    }

    init();
    return () => { cancelled = true; };
  }, [retryToken]);

  // A fresh result takes focus, so keyboard and screen-reader users land on it.
  useEffect(() => {
    if (result && !loading) document.getElementById("result-title")?.focus();
  }, [result, loading]);

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
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-10
                   focus:rounded-lg focus:bg-blue-600 focus:px-4 focus:py-2 focus:text-white"
      >
        Skip to content
      </a>
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">
              Complaint Triage
            </h1>
            <p className="text-xs text-gray-400 mt-0.5">
              Intelligent routing for consumer financial complaints
            </p>
            {/* A visitor must never think they filed a real complaint, or paste details they shouldn't. */}
            <p className="text-xs text-gray-300 mt-2 max-w-prose">
              A research demo trained on public CFPB complaints, not a CFPB service. Your text
              is analysed by the demo&rsquo;s model and not filed as a complaint anywhere.
              Please leave out personal details.
            </p>
          </div>
          {/* Backend status indicator: announced when it changes */}
          <div role="status" className="flex items-center gap-2 text-xs text-gray-300 shrink-0 pt-1.5">
            <span
              aria-hidden="true"
              className={`w-2 h-2 rounded-full ${
                backendStatus === "ready"
                  ? "bg-emerald-500"
                  : backendStatus === "error"
                  ? "bg-red-500"
                  : "bg-yellow-500"
              }`}
            />
            {backendStatus === "ready"
              ? "Model online"
              : backendStatus === "error"
              ? "Offline"
              : "Starting up…"}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main id="main" tabIndex={-1} className="max-w-7xl mx-auto px-6 py-8 focus:outline-none">
        {/* Show loading state if backend isn't ready yet.
            While waiting, and if it can't be reached, display stored results so
            visitors can still see what the system does. */}
        {backendStatus !== "ready" && (
          <>
            <LoadingState status={backendStatus} onRetry={handleRetry} />
            <section className="mt-8" aria-labelledby="samples-title">
              <div className="flex items-center gap-2 mb-6">
                <h2 id="samples-title" className="text-xs font-medium text-gray-300 uppercase tracking-wider">
                  {backendStatus === "error"
                    ? "Stored results from earlier runs"
                    : "Stored results while the system starts"}
                </h2>
                <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-500/15 text-blue-300">
                  Stored
                </span>
              </div>
              {/* Two stored results side by side: automated vs escalated */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Left: an example the model handles automatically */}
                <section className="space-y-4" aria-labelledby="sample-automatic">
                  <h3 id="sample-automatic" className="text-xs text-blue-300 font-medium uppercase tracking-wider">
                    {CACHED_EXAMPLE_LABEL}: handled automatically
                  </h3>
                  <ResultsDashboard result={CACHED_EXAMPLE} headingLevel="h4" />
                  <AgenticTrace trace={CACHED_EXAMPLE.agentic_trace} headingLevel="h4" />
                </section>
                {/* Right: an example the model sends to a person */}
                <section className="space-y-4" aria-labelledby="sample-escalation">
                  <h3 id="sample-escalation" className="text-xs text-amber-300 font-medium uppercase tracking-wider">
                    {CACHED_ESCALATION_LABEL}: goes to a person
                  </h3>
                  <ResultsDashboard result={CACHED_ESCALATION} headingLevel="h4" />
                  <AgenticTrace trace={CACHED_ESCALATION.agentic_trace} headingLevel="h4" />
                </section>
              </div>
            </section>
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
                <p role="alert" className="text-red-400 text-sm">{backendError}</p>
              )}

              {/* Empty state — before any classification */}
              {!result && !loading && (
                <div className="flex items-center justify-center h-64 border border-dashed border-gray-700 rounded-xl">
                  <p className="text-gray-400 text-sm text-center px-6">
                    Submit a complaint or try an example to see how it&rsquo;s routed
                  </p>
                </div>
              )}

              {/* Loading state for individual requests (announced) */}
              {loading && (
                <div role="status" className="flex items-center justify-center h-64">
                  <div className="flex items-center gap-3 text-gray-300">
                    <svg className="motion-safe:animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none" aria-hidden="true">
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
                      type="button"
                      aria-pressed={showTech}
                      onClick={() => setShowTech((s) => !s)}
                      className="text-xs px-3 py-1.5 rounded-full border transition-colors duration-150
                        border-gray-600 text-gray-300 hover:text-white hover:border-gray-400
                        aria-pressed:border-blue-400 aria-pressed:text-blue-200"
                    >
                      Technical details
                    </button>
                  </div>
                  <ResultsDashboard result={result} showTech={showTech} headingId="result-title" />
                </div>
              )}
            </div>
          </div>
        )}

        {/* The explanation flows below the opening row, so the input and the result sit
            side by side and neither column runs far past the first screen. */}
        {backendStatus === "ready" && result && !loading && (
          <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
            <AgenticTrace trace={result.agentic_trace} showTech={showTech} />
            <RagContext passages={result.rag_context} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 px-6 py-4 mt-auto">
        <p className="max-w-7xl mx-auto text-center text-xs text-gray-400">
          Built on the CFPB Consumer Complaint Database ·{" "}
          <a
            href="https://github.com/Nduka99/customer_complaint_triage"
            className="text-blue-300 underline underline-offset-2 hover:text-blue-200"
          >
            Source on GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
