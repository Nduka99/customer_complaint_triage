import { useState, useEffect } from "react";

/**
 * AgenticTrace — Animates the 6-step pipeline trace to show the system "thinking".
 *
 * The backend returns agentic_trace as an array of 6 strings:
 *   1. Models loaded from HF Hub (cached).
 *   2. RoBERTa-D inference: <label> (<conf>)
 *   3. ModernBERT inference: <label> (<conf>)
 *   4. LR Stacker decision: <label> (<conf>)
 *   5. Thompson Sampling: Arm <n> (<decision>) selected
 *   6. RAG retrieved <n> passages for '<label>'
 *
 * Each step appears with a staggered delay (200ms apart), creating the
 * impression of the pipeline executing in real time. This is a deliberate
 * HCI choice — it makes the agentic architecture tangible and explorable
 * rather than presenting results as a black box.
 */

// Map step index to an icon that represents the pipeline stage
const STEP_ICONS = ["🔧", "🤖", "🤖", "📊", "🎰", "📚"];

// Map step index to a subtle color accent for visual grouping
// Steps 1-3 (inference) = blue, Step 4 (ensemble) = purple,
// Step 5 (bandit) = amber, Step 6 (RAG) = teal
const STEP_COLORS = [
  "border-gray-600/50",   // 1. Model loading — neutral
  "border-blue-600/50",   // 2. RoBERTa-D — blue (inference)
  "border-blue-600/50",   // 3. ModernBERT — blue (inference)
  "border-purple-600/50", // 4. LR Stacker — purple (ensemble)
  "border-amber-600/50",  // 5. Thompson Sampling — amber (RL)
  "border-teal-600/50",   // 6. RAG retrieval — teal (retrieval)
];

export default function AgenticTrace({ trace }) {
  // visibleCount controls how many steps are revealed — drives the animation
  const [visibleCount, setVisibleCount] = useState(0);

  useEffect(() => {
    if (!trace || trace.length === 0) return;

    // Reset when a new trace arrives (new classification request)
    setVisibleCount(0);

    // Reveal one step every 200ms for a cascading animation effect
    const timers = trace.map((_, i) =>
      setTimeout(() => setVisibleCount(i + 1), (i + 1) * 200)
    );

    // Clean up timers if component unmounts or trace changes mid-animation
    return () => timers.forEach(clearTimeout);
  }, [trace]);

  if (!trace || trace.length === 0) return null;

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-100 tracking-tight mb-3">
        Pipeline Trace
      </h2>
      <div className="space-y-2">
        {trace.map((step, i) => (
          <div
            key={i}
            className={`
              flex items-start gap-3 px-3 py-2.5 rounded-lg
              bg-gray-900/40 border-l-2 ${STEP_COLORS[i] || "border-gray-600/50"}
              transition-all duration-300 ease-out
              ${i < visibleCount
                ? "opacity-100 translate-x-0"
                : "opacity-0 -translate-x-4"
              }
            `}
          >
            {/* Step icon */}
            <span className="text-sm mt-0.5 shrink-0">
              {STEP_ICONS[i] || "▸"}
            </span>
            {/* Step text */}
            <p className="text-sm text-gray-300 font-mono leading-relaxed">
              {step}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
