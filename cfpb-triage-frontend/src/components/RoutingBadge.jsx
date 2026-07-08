/**
 * RoutingBadge — Shows how the complaint will be handled.
 *
 * Two possible states:
 *   1. Automated (arms 0-2): handled without human involvement. Green badge.
 *   2. Human review (arm 3): referred to a specialist. Amber badge — either the
 *      final ensemble confidence fell below the 0.55 gate (forced), the first
 *      model fell below the 0.65 early-exit gate, or the bandit itself picked
 *      the escalation arm from its posteriors.
 *
 * Layer 1 (default) uses plain language: what happens next and why, in terms
 * of historical resolution outcomes. Layer 2 (`showTech`) reveals the internal
 * strategy name and the raw routing reason from the backend (Thompson samples
 * or the threshold message).
 */

// Internal strategy names — only shown in the technical-details layer
// (matches pipeline.py arm_names)
const ARM_LABELS = {
  0: "RoBERTa-D direct",
  1: "ModernBERT direct",
  2: "Stacked ensemble",
  3: "Human escalation",
};

// Translate the backend's raw routing reason into plain product language.
// Forced escalations mention a threshold; bandit-chosen escalation does not.
function friendlyReason(routing) {
  if (routing.arm === 3) {
    if (/threshold/i.test(routing.reason)) {
      return "Confidence was too low for automated handling, so this complaint is queued for a specialist.";
    }
    return "For this type of complaint, human review has historically led to better resolutions.";
  }
  return "Chosen as the most reliable handling strategy for this complaint category, based on historical resolution outcomes.";
}

export default function RoutingBadge({ routing, showTech = false }) {
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
      {/* Top row: badge pill + plain-language outcome */}
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
          {isEscalated ? "Human review" : "Automated"}
        </span>
        <span className="text-sm text-gray-300">
          {isEscalated
            ? "Referred to a specialist"
            : "Handled automatically"}
        </span>
      </div>

      {/* Plain-language reason */}
      <p className={`mt-2 text-xs leading-relaxed ${
        isEscalated ? "text-amber-200/70" : "text-emerald-200/70"
      }`}>
        {friendlyReason(routing)}
      </p>

      {/* Technical layer: internal strategy name + raw routing reason
          (Thompson samples or the threshold message from the backend) */}
      {showTech && (
        <p className="mt-2 text-xs text-gray-500 font-mono leading-relaxed">
          Strategy: {ARM_LABELS[routing.arm]} · {routing.reason}
        </p>
      )}
    </div>
  );
}
