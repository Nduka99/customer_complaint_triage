import RoutingBadge from "./RoutingBadge";

/**
 * ResultsDashboard — Displays classification result, confidence meter, and routing decision.
 *
 * This is the primary output panel. It receives the full backend response and
 * renders three sections:
 *   1. Product classification label (one of 10 CFPB categories)
 *   2. Confidence bar with color coding (red < 0.55, yellow 0.55-0.80, green > 0.80)
 *   3. RoutingBadge showing the Thompson Sampling bandit decision
 *
 * The 0.55 threshold for the confidence bar color matches the NB11 optimal
 * escalation threshold — visually reinforcing the RL routing logic.
 */

// Confidence → color mapping aligned with NB11 threshold (0.55)
function getConfidenceColor(confidence) {
  if (confidence < 0.55) return { bar: "bg-red-500", text: "text-red-400", label: "Low" };
  if (confidence < 0.80) return { bar: "bg-yellow-500", text: "text-yellow-400", label: "Medium" };
  return { bar: "bg-emerald-500", text: "text-emerald-400", label: "High" };
}

export default function ResultsDashboard({ result }) {
  if (!result) return null;

  const { classification, routing } = result;
  const confidence = classification.confidence;
  const pct = (confidence * 100).toFixed(1);
  const color = getConfidenceColor(confidence);

  return (
    <div className="space-y-4">
      {/* Section header */}
      <h2 className="text-lg font-semibold text-gray-100 tracking-tight">
        Triage Result
      </h2>

      {/* Classification card */}
      <div className="bg-gray-900/60 border border-gray-700/50 rounded-xl p-4 space-y-4">
        {/* Product label */}
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-1">
            Predicted Product
          </p>
          <p className="text-xl font-semibold text-white leading-snug">
            {classification.label}
          </p>
        </div>

        {/* Confidence meter */}
        <div>
          <div className="flex items-baseline justify-between mb-1.5">
            <p className="text-xs text-gray-500 uppercase tracking-wider font-medium">
              Confidence
            </p>
            <span className={`text-sm font-mono font-semibold ${color.text}`}>
              {pct}%
              <span className="text-xs font-normal ml-1 opacity-70">{color.label}</span>
            </span>
          </div>
          {/* Bar container */}
          <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
            {/* Filled bar — width set by confidence percentage */}
            <div
              className={`h-full rounded-full ${color.bar} transition-all duration-700 ease-out`}
              style={{ width: `${pct}%` }}
            />
          </div>
          {/* Threshold marker label */}
          <div className="flex justify-between mt-1">
            <span className="text-[10px] text-gray-600">0%</span>
            <span className="text-[10px] text-gray-600">
              Escalation threshold: 55%
            </span>
            <span className="text-[10px] text-gray-600">100%</span>
          </div>
        </div>
      </div>

      {/* Routing decision — the RL bandit component */}
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-2">
          Routing Decision
        </p>
        <RoutingBadge routing={routing} />
      </div>
    </div>
  );
}
