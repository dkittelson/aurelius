import { useDashboardStats, useGraphStats } from "../hooks/useAurelius";
import StatCard from "../components/StatCard";
import RiskDistributionChart from "../components/RiskDistributionChart";

export default function Dashboard() {
  const { data: dash, isLoading: dashLoading } = useDashboardStats();
  const { data: graph } = useGraphStats();

  if (dashLoading) {
    return (
      <div className="flex h-full items-center justify-center text-zinc-400">
        Loading dashboard…
      </div>
    );
  }

  if (!dash) {
    return (
      <div className="flex h-full items-center justify-center text-zinc-500">
        Could not connect to the Aurelius API. Make sure the backend is running on port 8000.
      </div>
    );
  }

  const illicitPct = graph
    ? (graph.illicit_rate * 100).toFixed(2) + "% illicit rate"
    : undefined;

  return (
    <div className="flex flex-col gap-6 p-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Overview</h1>
        <p className="text-sm text-zinc-400">
          Dataset: <span className="font-mono text-zinc-200">{dash.dataset}</span>
          {dash.model_val_auprc != null && (
            <> · Model val AUPRC: <span className="font-mono text-zinc-200">{dash.model_val_auprc.toFixed(4)}</span></>
          )}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Total Nodes" value={dash.total_nodes.toLocaleString()} />
        <StatCard label="Total Edges" value={dash.total_edges.toLocaleString()} />
        <StatCard
          label="Flagged Nodes"
          value={dash.flagged_nodes.toLocaleString()}
          sub={illicitPct}
          color="orange"
        />
        <StatCard
          label="Critical Clusters"
          value={dash.critical_clusters}
          color={dash.critical_clusters > 0 ? "red" : "default"}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-5">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-zinc-400">
            Cluster Risk Distribution
          </h2>
          <RiskDistributionChart
            critical={dash.critical_clusters}
            high={dash.high_clusters}
            medium={dash.medium_clusters}
            low={dash.low_clusters}
          />
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-5">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-zinc-400">
            Graph Summary
          </h2>
          {graph ? (
            <div className="flex flex-col gap-3 text-sm">
              {[
                ["Illicit nodes", graph.num_illicit.toLocaleString()],
                ["Licit nodes", graph.num_licit.toLocaleString()],
                ["Unknown nodes", graph.num_unknown.toLocaleString()],
                ["Illicit rate (labeled)", (graph.illicit_rate * 100).toFixed(2) + "%"],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-zinc-400">{k}</span>
                  <span className="font-mono text-zinc-200">{v}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-zinc-500">Loading graph stats…</p>
          )}
        </div>
      </div>
    </div>
  );
}
