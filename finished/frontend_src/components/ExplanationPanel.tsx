import type { NodeExplanation } from "../api/client";

interface Props {
  explanation: NodeExplanation | null;
  loading?: boolean;
}

export default function ExplanationPanel({ explanation, loading }: Props) {
  if (loading) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
        <p className="text-sm text-zinc-400 animate-pulse">
          Generating explanation...
        </p>
      </div>
    );
  }

  if (!explanation) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
        <p className="text-sm text-zinc-500">
          Click a node to generate an explanation.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4 space-y-4">
      <h3 className="text-sm font-semibold text-zinc-200">
        Explanation — Node {explanation.node_id}
      </h3>

      {/* Important Edges */}
      <div>
        <h4 className="text-xs font-medium text-zinc-400 mb-2">
          Top Edges (Transactions)
        </h4>
        {explanation.important_edges.length > 0 ? (
          <ul className="space-y-1">
            {explanation.important_edges.slice(0, 5).map((e, i) => (
              <li
                key={i}
                className="flex items-center justify-between text-xs text-zinc-300"
              >
                <span>
                  {e.src} → {e.dst}
                </span>
                <span className="rounded bg-zinc-800 px-1.5 py-0.5">
                  {(e.importance * 100).toFixed(1)}%
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-zinc-500">No significant edges</p>
        )}
      </div>

      {/* Important Features */}
      <div>
        <h4 className="text-xs font-medium text-zinc-400 mb-2">
          Top Features
        </h4>
        {explanation.important_features.length > 0 ? (
          <div className="space-y-1">
            {explanation.important_features.slice(0, 5).map((f, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs text-zinc-300 w-16 shrink-0">
                  Feat {f.feature_index}
                </span>
                <div className="flex-1 h-2 rounded-full bg-zinc-800 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-blue-500"
                    style={{
                      width: `${Math.min(Math.abs(f.importance) * 100, 100)}%`,
                    }}
                  />
                </div>
                <span className="text-xs text-zinc-500 w-12 text-right">
                  {f.importance.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-zinc-500">No significant features</p>
        )}
      </div>
    </div>
  );
}
