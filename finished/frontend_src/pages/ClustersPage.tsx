import { useState } from "react";
import { useClusters, useSubgraph, useInvestigate } from "../hooks/useAurelius";
import type { ClusterSummary } from "../api/client";
import ClusterTable from "../components/ClusterTable";
import GraphView from "../components/GraphView";
import ForensicReport from "../components/ForensicReport";

export default function ClustersPage() {
  const [threshold, setThreshold] = useState(0.75);
  const [selected, setSelected] = useState<ClusterSummary | null>(null);
  const [report, setReport] = useState<import("../api/client").InvestigateResponse | null>(null);

  const { data: clustersData, isLoading: clustersLoading } = useClusters(threshold, 2, 15);
  const { data: subgraph } = useSubgraph(selected?.node_ids ?? []);
  const investigate = useInvestigate();

  const handleInvestigate = async () => {
    if (!selected) return;
    try {
      const result = await investigate.mutateAsync({
        clusterId: selected.cluster_id,
        nodeIds: selected.node_ids,
      });
      setReport(result);
    } catch {
      // handled below via investigate.error
    }
  };

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <div className="flex items-center gap-4">
        <h1 className="text-2xl font-bold text-white">Suspicious Clusters</h1>
        <div className="ml-auto flex items-center gap-3 text-sm">
          <label className="text-zinc-400">
            Threshold:{" "}
            <span className="font-mono text-zinc-200">{threshold.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0.5"
            max="0.99"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="accent-blue-500"
          />
        </div>
      </div>

      <div className="grid flex-1 grid-cols-1 gap-4 overflow-hidden lg:grid-cols-2">
        {/* Left: cluster list */}
        <div className="flex flex-col gap-4 overflow-auto rounded-xl border border-zinc-800 bg-zinc-900 p-4">
          <ClusterTable
            clusters={clustersData?.clusters ?? []}
            onSelect={(c) => {
              setSelected(c);
              setReport(null);
            }}
            selectedId={selected?.cluster_id}
            loading={clustersLoading}
          />
        </div>

        {/* Right: graph + report */}
        <div className="flex flex-col gap-4 overflow-auto">
          {selected ? (
            <>
              <div className="relative h-72 rounded-xl border border-zinc-800 bg-zinc-950">
                {subgraph ? (
                  <GraphView nodes={subgraph.nodes} edges={subgraph.edges} />
                ) : (
                  <div className="flex h-full items-center justify-center text-zinc-500 text-sm">
                    Loading subgraph…
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-zinc-300">
                    Cluster {selected.cluster_id} — {selected.num_nodes} nodes
                  </h2>
                  <button
                    onClick={handleInvestigate}
                    disabled={investigate.isPending}
                    className="rounded-lg bg-blue-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-blue-500 disabled:opacity-50 transition-colors"
                  >
                    {investigate.isPending ? "Investigating…" : "Investigate with AI"}
                  </button>
                </div>
                <ForensicReport
                  report={report}
                  loading={investigate.isPending}
                  error={
                    investigate.isError
                      ? (investigate.error as Error).message
                      : null
                  }
                />
              </div>
            </>
          ) : (
            <div className="flex h-full items-center justify-center text-zinc-500 text-sm">
              Select a cluster from the list to view its graph and investigate.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
