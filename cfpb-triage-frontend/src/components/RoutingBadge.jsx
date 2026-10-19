/**
 * RoutingBadge — Shows how the complaint will be handled.
 *
 * Two possible states:
 *   1. Automated (arms 0-2): handled without human involvement. Blue badge.
 *   2. Human review (arm 3): would go to a person. Amber badge — either the
 *      final ensemble confidence fell below the 0.55 gate (forced), the first
 *      model fell below the 0.65 early-exit gate, or the bandit itself picked
 *      the escalation arm from its posteriors.
 *
 * Layer 1 (default) uses plain language: what would happen and which check
 * decided it. Layer 2 (`showTech`) reveals the internal
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

// Translate the backend's raw routing reason into plain product language. It says
// "would" because this is a demo: nothing is queued. The reason strings name the
// confidence gate that fired (pipeline.py); above both gates the route is a random
// draw from what past outcomes taught the router.
function friendlyReason(routing) {
  if (routing.arm !== 3) {
    return "Both confidence checks passed, and this time the router chose to handle the complaint automatically.";
  }
  if (routing.reason.includes("early-exit")) {
    return "The first model was less than 65% sure of the category, so this complaint would go to a person.";
  }
  if (routing.reason.includes("below 0.55")) {
    return "The combined confidence was below 55%, so this complaint would go to a person.";
  }
  return "The router's estimate from past outcomes pointed to a person this time, so it would send this complaint to one.";
}

export default function RoutingBadge({ routing, showTech = false }) {
  if (!routing) return null;

  const isEscalated = routing.arm === 3;

  return (
    <div
      className={`rounded-xl border px-4 py-3 ${
        isEscalated
          ? "bg-amber-950/30 border-amber-700/40"
          : "bg-blue-950/30 border-blue-700/40"
      }`}
    >
      {/* Top row: badge pill + plain-language outcome */}
      <div className="flex items-center gap-2.5">
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold tracking-wide uppercase ${
            isEscalated
              ? "bg-amber-500/20 text-amber-400"
              : "bg-blue-500/20 text-blue-300"
          }`}
        >
          {/* Status dot (decorative: the badge text carries the status) */}
          <span
            aria-hidden="true"
            className={`w-1.5 h-1.5 rounded-full ${
              isEscalated ? "bg-amber-400" : "bg-blue-300"
            }`}
          />
          {isEscalated ? "Human review" : "Automated"}
        </span>
        <span className="text-sm text-gray-300">
          {isEscalated
            ? "Would go to a person"
            : "Would be handled automatically"}
        </span>
      </div>

      {/* Plain-language reason */}
      <p className={`mt-2 text-xs leading-relaxed max-w-prose ${
        isEscalated ? "text-amber-200" : "text-blue-100"
      }`}>
        {friendlyReason(routing)}
      </p>

      {/* Technical layer: internal strategy name + raw routing reason
          (Thompson samples or the threshold message from the backend) */}
      {showTech && (
        <p className="mt-2 text-xs text-gray-300 font-mono leading-relaxed max-w-prose">
          Strategy: {ARM_LABELS[routing.arm]} · {routing.reason}
        </p>
      )}
    </div>
  );
}
