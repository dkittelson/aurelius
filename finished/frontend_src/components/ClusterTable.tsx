import type { ClusterSummary } from "../api/client";
import RiskBadge from "./RiskBadge";

interface Props {
  clusters: ClusterSummary[];
  onSelect: (cluster: ClusterSummary) => void;
  selectedId?: number;
  loading?: boolean;
}

export default function ClusterTable({ clusters, onSelect, selectedId, loading }: Props) {
  if (loading) {
    return (
      <div className="flex h-40 items-center justify-center text-zinc-500">
        Detecting clusters…
      </div>
    );
  }

  if (clusters.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center text-zinc-500">
        No suspicious clusters detected at current threshold.
      </div>
    );
  }

  return (
    <div className="overflow-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 text-left text-xs uppercase tracking-wider text-zinc-500">
            <th className="px-3 py-2">ID</th>
            <th className="px-3 py-2">Risk</th>
            <th className="px-3 py-2">Nodes</th>
            <th className="px-3 py-2">Avg p</th>
            <th className="px-3 py-2">Max p</th>
            <th className="px-3 py-2">Density</th>
          </tr>
        </thead>
        <tbody>
          {clusters.map((c) => (
            <tr
              key={c.cluster_id}
              onClick={() => onSelect(c)}
              className={`cursor-pointer border-b border-zinc-900 transition-colors hover:bg-zinc-800 ${
                selectedId === c.cluster_id ? "bg-zinc-800" : ""
              }`}
            >
              <td className="px-3 py-2 font-mono text-zinc-300">{c.cluster_id}</td>
              <td className="px-3 py-2">
                <RiskBadge level={c.risk_level} />
              </td>
              <td className="px-3 py-2 text-zinc-300">{c.num_nodes}</td>
              <td className="px-3 py-2 font-mono text-zinc-300">
                {(c.avg_confidence * 100).toFixed(1)}%
              </td>
              <td className="px-3 py-2 font-mono text-zinc-300">
                {(c.max_confidence * 100).toFixed(1)}%
              </td>
              <td className="px-3 py-2 font-mono text-zinc-400">
                {c.density.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
