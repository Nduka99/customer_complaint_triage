import RoutingBadge from "./RoutingBadge";

/**
 * ResultsDashboard — Displays classification result, confidence meter, and routing decision.
 *
 * This is the primary output panel. It receives the full backend response and
 * renders three sections:
 *   1. Complaint category label (one of 10 CFPB product categories)
 *   2. Confidence bar with qualitative color coding (Low / Medium / High)
 *   3. RoutingBadge showing how the complaint will be handled
 *
 * Note on thresholds: the pipeline has TWO automation gates — a 0.65 early-exit
 * gate on the first model's confidence and a 0.55 escalation gate on the final
 * ensemble confidence. Because the bar displays whichever confidence the result
 * carries, we deliberately do NOT draw a single numeric threshold marker on it
 * (it would be wrong for one of the two paths). The exact gates are shown in
 * the technical-details layer instead.
 */

// Confidence → qualitative color band. The 0.65 low-band boundary mirrors the
// early-exit gate so "Low" visually aligns with results likely to be reviewed.
// The number itself stays neutral text; the band shows on the bar and in words.
function getConfidenceColor(confidence) {
  if (confidence < 0.65) return { bar: "bg-red-500", label: "Low" };
  if (confidence < 0.80) return { bar: "bg-yellow-500", label: "Medium" };
  return { bar: "bg-emerald-500", label: "High" };
}

// `headingLevel` lets stored samples nest under their own section heading;
// `headingId` lets the app move focus to a fresh result.
export default function ResultsDashboard({
  result,
  showTech = false,
  headingLevel = "h2",
  headingId,
}) {
  if (!result) return null;

  const { classification, routing } = result;
  const confidence = classification.confidence;
  const pct = (confidence * 100).toFixed(1);
  const color = getConfidenceColor(confidence);
  const Heading = headingLevel;

  return (
    <div className="space-y-4">
      {/* Section header */}
      <Heading
        id={headingId}
        tabIndex={headingId ? -1 : undefined}
        className="text-lg font-semibold text-gray-100 tracking-tight"
      >
        Result
      </Heading>

      {/* Classification card */}
      <div className="bg-gray-900/60 border border-gray-700 rounded-xl p-4 space-y-4">
        {/* Category label */}
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-wider font-medium mb-1">
            Complaint category
          </p>
          <p className="text-xl font-semibold text-white leading-snug">
            {classification.label}
          </p>
        </div>

        {/* Confidence meter */}
        <div>
          <div className="flex items-baseline justify-between mb-1.5">
            <p className="text-xs text-gray-400 uppercase tracking-wider font-medium">
              Confidence
            </p>
            <span className="text-sm font-mono font-semibold text-gray-100">
              {pct}%
              <span className="text-xs font-normal ml-1 text-gray-300">{color.label}</span>
            </span>
          </div>
          {/* Bar container: a meter, so screen readers hear the value, not just a shape */}
          <div
            role="meter"
            aria-label="Confidence"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Number(pct)}
            aria-valuetext={`${pct}% (${color.label})`}
            className="w-full h-2 bg-gray-800 rounded-full overflow-hidden"
          >
            {/* Filled bar — width set by confidence percentage */}
            <div
              className={`h-full rounded-full ${color.bar} motion-safe:transition-all motion-safe:duration-700 ease-out`}
              style={{ width: `${pct}%` }}
            />
          </div>
          {/* Technical layer: exact automation gates from pipeline.py */}
          {showTech && (
            <p className="mt-1.5 text-xs text-gray-400 font-mono max-w-prose">
              Automation gates: first-pass confidence ≥ 0.65 (early-exit) ·
              final ensemble confidence ≥ 0.55 (routing)
            </p>
          )}
        </div>
      </div>

      {/* Routing decision */}
      <div>
        <p className="text-xs text-gray-400 uppercase tracking-wider font-medium mb-2">
          Routing
        </p>
        <RoutingBadge routing={routing} showTech={showTech} />
      </div>
    </div>
  );
}
