import { useState } from "react";
import { useTopK } from "../hooks/useAurelius";
import RiskBadge from "../components/RiskBadge";

export default function TopFlaggedPage() {
  const [k, setK] = useState(50);
  const [threshold, setThreshold] = useState(0.5);
  const { data, isLoading } = useTopK(k, threshold);

  return (
    <div className="flex flex-col gap-6 p-6">
      <div className="flex flex-wrap items-center gap-4">
        <h1 className="text-2xl font-bold text-white">Top Flagged Nodes</h1>

        <div className="ml-auto flex flex-wrap items-center gap-4 text-sm">
          <label className="text-zinc-400">
            Show top{" "}
            <select
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="ml-1 rounded bg-zinc-800 px-2 py-1 text-zinc-200"
            >
              {[20, 50, 100, 200].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </label>

          <label className="text-zinc-400">
            Min probability:{" "}
            <span className="font-mono text-zinc-200">{threshold.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0.0"
            max="0.99"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="accent-blue-500"
          />
        </div>
      </div>

      {data && (
        <p className="text-sm text-zinc-400">
          {data.total_above_threshold.toLocaleString()} nodes exceed threshold{" "}
          <span className="font-mono">{threshold.toFixed(2)}</span>. Showing top{" "}
          {data.nodes.length}.
        </p>
      )}

      <div className="overflow-auto rounded-xl border border-zinc-800 bg-zinc-900">
        {isLoading ? (
          <div className="flex h-40 items-center justify-center text-zinc-500">
            Loading predictions…
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left text-xs uppercase tracking-wider text-zinc-500">
                <th className="px-4 py-3">Rank</th>
                <th className="px-4 py-3">Node ID</th>
                <th className="px-4 py-3">Risk</th>
                <th className="px-4 py-3">Illicit Probability</th>
                <th className="px-4 py-3">Confidence Bar</th>
              </tr>
            </thead>
            <tbody>
              {(data?.nodes ?? []).map((node, i) => (
                <tr
                  key={node.node_id}
                  className="border-b border-zinc-900 hover:bg-zinc-800 transition-colors"
                >
                  <td className="px-4 py-2 text-zinc-500">#{i + 1}</td>
                  <td className="px-4 py-2 font-mono text-zinc-200">{node.node_id}</td>
                  <td className="px-4 py-2">
                    <RiskBadge level={node.risk_level} />
                  </td>
                  <td className="px-4 py-2 font-mono text-zinc-300">
                    {(node.illicit_prob * 100).toFixed(2)}%
                  </td>
                  <td className="px-4 py-2 w-40">
                    <div className="h-2 w-full rounded-full bg-zinc-800">
                      <div
                        className="h-2 rounded-full transition-all"
                        style={{
                          width: `${node.illicit_prob * 100}%`,
                          background:
                            node.illicit_prob >= 0.9
                              ? "#ef4444"
                              : node.illicit_prob >= 0.75
                              ? "#f97316"
                              : node.illicit_prob >= 0.6
                              ? "#eab308"
                              : "#22c55e",
                        }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
