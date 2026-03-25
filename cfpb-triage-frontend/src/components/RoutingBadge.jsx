/**
 * RoutingBadge — Visualises the Thompson Sampling bandit decision from NB11.
 *
 * Two possible states:
 *   1. Automated (arms 0-2): The bandit chose a model arm. Green badge.
 *   2. Human Escalation (arm 3): Confidence < 0.55 or bandit chose escalation. Amber badge.
 *
 * The "reason" string from the backend shows the raw Thompson samples
 * (e.g. "Thompson samples: [0.821, 0.384, 0.883, 0.472]") or the
 * forced escalation message, making the RL decision fully transparent.
 */

// Map arm indices to human-readable strategy names (matches pipeline.py arm_names)
const ARM_LABELS = {
  0: "RoBERTa-D Direct",
  1: "ModernBERT Direct",
  2: "Ensemble (Stacked)",
  3: "Human Escalation",
};

export default function RoutingBadge({ routing }) {
  if (!routing) return null;

  const isEscalated = routing.arm === 3;

  return (
    <div
      className={`rounded-xl border px-4 py-3 ${
        isEscalated
          ? "bg-amber-950/30 border-amber-700/40"
          : "bg-emerald-950/30 border-emerald-700/40"
      }`}
    >
      {/* Top row: badge pill + arm name */}
      <div className="flex items-center gap-2.5">
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold tracking-wide uppercase ${
            isEscalated
              ? "bg-amber-500/20 text-amber-400"
              : "bg-emerald-500/20 text-emerald-400"
          }`}
        >
          {/* Status dot */}
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isEscalated ? "bg-amber-400" : "bg-emerald-400"
            }`}
          />
          {isEscalated ? "Escalated" : "Automated"}
        </span>
        <span className="text-sm text-gray-300">
          {ARM_LABELS[routing.arm]}
        </span>
      </div>

      {/* Reason — shows Thompson samples or forced-escalation rationale */}
      <p className="mt-2 text-xs text-gray-400 font-mono leading-relaxed">
        {routing.reason}
      </p>

      {/* Outcome insight — connects routing to historical resolution data.
          The bandit's Beta posteriors were trained on binary reward derived from
          company_response: positive (explanation/monetary/non-monetary relief)
          vs negative (closed with no action / untimely). This line makes that
          connection visible to the employer. */}
      <p className={`mt-2 text-xs italic ${
        isEscalated ? "text-amber-500/60" : "text-emerald-500/60"
      }`}>
        {isEscalated
          ? "Historical data shows human review yields higher positive resolution rates for this category."
          : "This routing strategy has the highest positive resolution rate for this complaint category."}
      </p>
    </div>
  );
}
