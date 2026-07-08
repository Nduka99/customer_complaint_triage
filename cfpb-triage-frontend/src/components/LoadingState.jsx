import { useState, useEffect } from "react";

/**
 * LoadingState — Handles the backend cold start gracefully.
 *
 * The analysis service sleeps after long inactivity (free-tier hosting).
 * When a visitor arrives while it's asleep, the first connection can take
 * a minute or two while the service wakes, loads its models, and prepares
 * the regulation search index.
 *
 * This component turns that wait into a calm, professional experience:
 *   - A clear plain-language explanation of what's happening
 *   - An elapsed timer so the user knows progress is being made
 *   - Startup stages that light up over time
 *   - A "Try again" button when the connection fails (via onRetry)
 *
 * Props:
 *   status:  "connecting" | "loading" | "ready" | "error"
 *   onRetry: called when the user clicks "Try again" after an error
 */

// Startup stages shown during loading — plain language, no internals
const STARTUP_STAGES = [
  { label: "Starting the service", detail: "Waking up after inactivity" },
  { label: "Loading analysis models", detail: "Preparing the complaint classifiers" },
  { label: "Loading the knowledge base", detail: "CFPB regulation excerpts" },
  { label: "Preparing search", detail: "Indexing regulation content" },
];

export default function LoadingState({ status, onRetry }) {
  // Elapsed timer — counts up while loading so the user sees progress
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (status !== "connecting" && status !== "loading") return;

    // Compute elapsed from a start timestamp inside the interval callback —
    // no synchronous setState in the effect body (avoids cascading renders).
    const start = Date.now();
    const interval = setInterval(
      () => setElapsed(Math.floor((Date.now() - start) / 1000)),
      1000
    );
    return () => clearInterval(interval);
  }, [status]);

  // Don't render anything when the backend is ready
  if (status === "ready") return null;

  // Format elapsed time as M:SS
  const minutes = Math.floor(elapsed / 60);
  const seconds = elapsed % 60;
  const timeStr = `${minutes}:${seconds.toString().padStart(2, "0")}`;

  return (
    <div className="flex flex-col items-center justify-center py-12 px-6">
      {/* Spinner */}
      {status !== "error" && (
        <div className="relative mb-6">
          <div className="w-12 h-12 rounded-full border-2 border-gray-700" />
          <div className="absolute inset-0 w-12 h-12 rounded-full border-2 border-t-blue-500 animate-spin" />
        </div>
      )}

      {/* Status message */}
      {status === "error" ? (
        <>
          <p className="text-red-400 font-medium mb-2">Connection problem</p>
          <p className="text-sm text-gray-400 text-center max-w-md">
            We couldn&rsquo;t reach the analysis service. It may be restarting —
            please try again in a moment.
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-4 px-4 py-2 rounded-lg text-sm font-medium
                         bg-gray-800 text-gray-200 border border-gray-700
                         hover:bg-gray-700 hover:border-gray-600
                         transition-colors duration-150"
            >
              Try again
            </button>
          )}
        </>
      ) : (
        <>
          <p className="text-gray-200 font-medium mb-1">
            Starting the system…
          </p>
          <p className="text-sm text-gray-400 text-center max-w-md mb-4">
            The service is waking up after a period of inactivity. This usually
            takes a minute or two — after that, results arrive in about a second.
          </p>

          {/* Elapsed timer */}
          <span className="text-xs font-mono text-gray-500 mb-6">
            {timeStr} elapsed
          </span>

          {/* Startup stages — light up progressively over the expected ~90s */}
          <div className="w-full max-w-xs space-y-2">
            {STARTUP_STAGES.map((stage, i) => {
              // Each stage "starts" after a proportional fraction of the
              // expected load time (~90s) — an honest approximation, since
              // the backend doesn't stream real progress events.
              const stageDelay = (i / STARTUP_STAGES.length) * 90;
              const isActive = elapsed >= stageDelay;

              return (
                <div
                  key={stage.label}
                  className={`flex items-center gap-3 text-sm transition-all duration-500
                    ${isActive ? "opacity-100" : "opacity-20"}`}
                >
                  {/* Status indicator */}
                  <span
                    className={`w-2 h-2 rounded-full shrink-0 transition-colors duration-500
                      ${isActive ? "bg-blue-500" : "bg-gray-700"}`}
                  />
                  <div>
                    <span className={`${isActive ? "text-gray-200" : "text-gray-600"}`}>
                      {stage.label}
                    </span>
                    <span className="text-gray-600 ml-1.5 text-xs">
                      {stage.detail}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
