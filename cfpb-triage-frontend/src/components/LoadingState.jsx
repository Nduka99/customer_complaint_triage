import { useState, useEffect } from "react";
import { getWakeStep } from "../api";

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
 *   - Wake-up steps taken from what the Hugging Face Hub reports, not a guess
 *   - A "Try again" button when the connection fails (via onRetry)
 *
 * Props:
 *   status:  "connecting" | "loading" | "ready" | "error"
 *   onRetry: called when the user clicks "Try again" after an error
 */

// One label per wake step returned by getWakeStep (0 asleep, 1 starting, 2 running).
const WAKE_STEP_LABELS = [
  "Waking the service",
  "Loading the models and regulation index",
  "Connecting",
];
const WAKE_POLL_MS = 5000;

export default function LoadingState({ status, onRetry }) {
  // Elapsed timer — counts up while loading so the user sees progress
  const [elapsed, setElapsed] = useState(0);
  // How far the Space is in waking up; null until the Hub has answered.
  const [wakeStep, setWakeStep] = useState(null);

  useEffect(() => {
    if (status !== "connecting") return;

    let cancelled = false;
    async function refresh() {
      const step = await getWakeStep();
      if (!cancelled) setWakeStep(step);
    }
    refresh();
    const interval = setInterval(refresh, WAKE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [status]);

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
      {/* Spinner (decorative: the text below says what is happening) */}
      {status !== "error" && (
        <div className="relative mb-6" aria-hidden="true">
          <div className="w-12 h-12 rounded-full border-2 border-gray-700" />
          <div className="absolute inset-0 w-12 h-12 rounded-full border-2 border-t-blue-500 motion-safe:animate-spin" />
        </div>
      )}

      {/* Status message */}
      {status === "error" ? (
        <div role="alert" className="flex flex-col items-center">
          <p className="text-red-400 font-medium mb-2">Connection problem</p>
          <p className="text-sm text-gray-300 text-center max-w-md">
            We couldn&rsquo;t reach the analysis service. It may be restarting, so
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
        </div>
      ) : (
        <>
          <div role="status" className="flex flex-col items-center">
            <p className="text-gray-200 font-medium mb-1">
              Starting the system…
            </p>
            <p className="text-sm text-gray-300 text-center max-w-md mb-4">
              The service sleeps when nobody has used it for a while. Waking it usually
              takes a minute or two; after that, results arrive in about a second.
            </p>
          </div>

          {/* Elapsed timer (outside the status region so it is not announced every second) */}
          <span className="text-xs font-mono text-gray-400 mb-6">
            {timeStr} elapsed
          </span>

          {/* Wake-up steps, as the Hub reports them (unknown counts as the first step) */}
          <ol className="w-full max-w-xs space-y-2">
            {WAKE_STEP_LABELS.map((label, i) => {
              const current = wakeStep ?? 0;
              const isDone = i < current;
              const isCurrent = i === current;

              return (
                <li
                  key={label}
                  aria-current={isCurrent ? "step" : undefined}
                  className="flex items-center gap-3 text-sm"
                >
                  {/* Status dot: filled once a step has started */}
                  <span
                    aria-hidden="true"
                    className={`w-2 h-2 rounded-full shrink-0 ${
                      isDone || isCurrent ? "bg-blue-500" : "border border-gray-500"
                    }`}
                  />
                  <span
                    className={
                      isCurrent ? "text-gray-100 font-medium" : isDone ? "text-gray-300" : "text-gray-400"
                    }
                  >
                    {label}
                    {isDone && <span className="sr-only"> (done)</span>}
                    {isCurrent && <span className="sr-only"> (in progress)</span>}
                  </span>
                </li>
              );
            })}
          </ol>
        </>
      )}
    </div>
  );
}
