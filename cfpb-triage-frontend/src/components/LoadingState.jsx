import { useState, useEffect } from "react";

/**
 * LoadingState — Handles the HF Spaces cold start gracefully.
 *
 * HF Spaces free tier sleeps after 48h of inactivity. When an employer
 * visits the demo and the Space is sleeping, the first request can take
 * 2-3 minutes while it:
 *   1. Wakes the container
 *   2. Downloads RoBERTa-D + ModernBERT from HF Hub (~1GB)
 *   3. Loads the stacker, knowledge base, and bandit state
 *   4. Pre-builds BM25 indices for 2,689 passages
 *
 * This component turns that wait into a professional experience instead
 * of a blank screen. It shows:
 *   - A clear explanation of what's happening (honest, not vague)
 *   - An elapsed timer so the user knows progress is being made
 *   - The pipeline architecture being loaded (builds anticipation)
 *
 * Props:
 *   status: "connecting" | "loading" | "ready" | "error"
 *   error: string (optional error message)
 */

// Pipeline stages shown during loading — educates the employer while they wait
const PIPELINE_STAGES = [
  { label: "Container wake-up", detail: "Initialising HF Spaces runtime" },
  { label: "RoBERTa-D (distilled)", detail: "125M parameter classifier" },
  { label: "ModernBERT", detail: "149M parameter classifier" },
  { label: "LR Stacker", detail: "Ensemble meta-learner" },
  { label: "Knowledge base", detail: "2,689 CFPB regulatory passages" },
  { label: "BM25 indices", detail: "Per-category retrieval indices" },
];

export default function LoadingState({ status, error }) {
  // Elapsed timer — counts up while loading so the user sees progress
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (status !== "connecting" && status !== "loading") return;

    setElapsed(0);
    const interval = setInterval(() => setElapsed((e) => e + 1), 1000);
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
          <p className="text-red-400 font-medium mb-2">Connection failed</p>
          <p className="text-sm text-gray-400 text-center max-w-md">
            {error || "Unable to reach the ML backend. The Space may be rebuilding. Please try again in a few minutes."}
          </p>
        </>
      ) : (
        <>
          <p className="text-gray-200 font-medium mb-1">
            {status === "connecting"
              ? "Connecting to ML pipeline..."
              : "Loading models into memory..."}
          </p>
          <p className="text-sm text-gray-400 text-center max-w-md mb-4">
            The pipeline initialises on first access as models are downloaded
            from Hugging Face Hub. This typically takes 1-2 minutes.
            Subsequent requests are instant.
          </p>

          {/* Elapsed timer */}
          <span className="text-xs font-mono text-gray-500 mb-6">
            Elapsed: {timeStr}
          </span>

          {/* Pipeline stages — reveals what's being loaded */}
          <div className="w-full max-w-xs space-y-2">
            {PIPELINE_STAGES.map((stage, i) => {
              // Animate stages appearing based on elapsed time
              // Each stage "starts" after a proportional fraction of expected load time (~90s)
              const stageDelay = (i / PIPELINE_STAGES.length) * 90;
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
