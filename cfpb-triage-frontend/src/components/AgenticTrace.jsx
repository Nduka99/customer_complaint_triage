import { useState, useEffect } from "react";

/**
 * AgenticTrace — Animates the pipeline trace to show how the decision was made.
 *
 * The backend returns agentic_trace as an array of raw log strings, e.g.:
 *   "2. RoBERTa-D inference: Debt collection (0.52)"
 *   "5. Thompson Sampling: Arm 2 (Ensemble (Stacked)) selected"
 *
 * Those strings are developer language, so we translate each one into a
 * plain-language step at render time (parseStep). Doing the mapping here —
 * rather than changing the backend — keeps the deployed HF Space untouched
 * and automatically covers the cached sample results too.
 *
 * Layer 2 (`showTech`) reveals the raw backend log line under each step.
 *
 * Each step appears with a staggered delay (200ms apart), creating the
 * impression of the pipeline executing in real time. This is a deliberate
 * HCI choice — it makes the multi-step decision tangible and explorable
 * rather than presenting results as a black box.
 */

// Translate one raw backend trace line into { title, detail, tone }.
// Patterns must stay in sync with the trace strings built in pipeline.py.
// Unrecognized lines fall through to a neutral step showing the raw text,
// so a future backend change degrades gracefully instead of breaking.
function parseStep(step) {
  let m;

  if (/Models loaded/i.test(step)) {
    return {
      title: "System ready",
      detail: "Models and regulation knowledge base loaded",
      tone: "neutral",
    };
  }
  if ((m = step.match(/RoBERTa-D inference: (.+) \(([\d.]+)\)/))) {
    return {
      title: "First analysis",
      detail: `${m[1]} — ${Math.round(m[2] * 100)}% confident`,
      tone: "analysis",
    };
  }
  if ((m = step.match(/ModernBERT inference: (.+) \(([\d.]+)\)/))) {
    return {
      title: "Second analysis",
      detail: `${m[1]} — ${Math.round(m[2] * 100)}% confident`,
      tone: "analysis",
    };
  }
  if (/Early-exit/i.test(step)) {
    return {
      title: "Confidence check",
      detail: "The first analysis wasn't confident enough for automated handling",
      tone: "review",
    };
  }
  if ((m = step.match(/LR Stacker decision: (.+) \(([\d.]+)\)/))) {
    return {
      title: "Combined decision",
      detail: `${m[1]} — ${Math.round(m[2] * 100)}% confident`,
      tone: "decision",
    };
  }
  if (/Forced escalation/i.test(step)) {
    return {
      title: "Referred for human review",
      detail: "Uncertainty is too high for automated processing",
      tone: "review",
    };
  }
  if ((m = step.match(/Thompson Sampling: Arm (\d+)/))) {
    return m[1] === "3"
      ? {
          title: "Routing decision",
          detail: "Referred to a human specialist",
          tone: "review",
        }
      : {
          title: "Routing decision",
          detail: "Routed for automated processing",
          tone: "routing",
        };
  }
  if ((m = step.match(/RAG retrieved (\d+) passages/))) {
    const n = Number(m[1]);
    return {
      title: "Regulation lookup",
      detail:
        n === 0
          ? "No matching regulation excerpts found"
          : `Found ${n} relevant regulation excerpt${n === 1 ? "" : "s"}`,
      tone: "retrieval",
    };
  }
  // Fallback for unrecognized trace lines
  return { title: "Processing", detail: step, tone: "neutral" };
}

// Subtle left-border accent per step type for visual grouping
const TONE_BORDER = {
  neutral: "border-gray-600/50",
  analysis: "border-blue-600/50",
  decision: "border-purple-600/50",
  routing: "border-emerald-600/50",
  review: "border-amber-600/50",
  retrieval: "border-teal-600/50",
};

export default function AgenticTrace({ trace, showTech = false }) {
  // visibleCount controls how many steps are revealed — drives the animation
  const [visibleCount, setVisibleCount] = useState(0);

  useEffect(() => {
    if (!trace || trace.length === 0) return;

    // All updates run inside timer callbacks (never synchronously in the
    // effect body) to avoid cascading renders. The 0ms timer resets the
    // reveal count when a new trace arrives; in practice the parent unmounts
    // this component between requests, so no flash is visible.
    const timers = [
      setTimeout(() => setVisibleCount(0), 0),
      // Reveal one step every 200ms for a cascading animation effect
      ...trace.map((_, i) =>
        setTimeout(() => setVisibleCount(i + 1), (i + 1) * 200)
      ),
    ];

    // Clean up timers if component unmounts or trace changes mid-animation
    return () => timers.forEach(clearTimeout);
  }, [trace]);

  if (!trace || trace.length === 0) return null;

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-100 tracking-tight mb-3">
        How this decision was made
      </h2>
      <div className="space-y-2">
        {trace.map((step, i) => {
          const parsed = parseStep(step);
          return (
            <div
              key={i}
              className={`
                flex items-start gap-3 px-3 py-2.5 rounded-lg
                bg-gray-900/40 border-l-2 ${TONE_BORDER[parsed.tone]}
                transition-all duration-300 ease-out
                ${i < visibleCount
                  ? "opacity-100 translate-x-0"
                  : "opacity-0 -translate-x-4"
                }
              `}
            >
              {/* Step number */}
              <span
                className="shrink-0 w-5 h-5 mt-0.5 rounded-full bg-gray-800 border border-gray-700
                           flex items-center justify-center text-[10px] font-medium text-gray-400"
              >
                {i + 1}
              </span>
              {/* Step text: plain-language title + detail */}
              <div className="min-w-0">
                <p className="text-sm text-gray-200 leading-relaxed">
                  <span className="font-medium">{parsed.title}</span>
                  {parsed.detail && (
                    <span className="text-gray-400"> — {parsed.detail}</span>
                  )}
                </p>
                {/* Technical layer: the raw backend log line */}
                {showTech && (
                  <p className="mt-1 text-xs text-gray-600 font-mono leading-relaxed break-words">
                    {step}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
