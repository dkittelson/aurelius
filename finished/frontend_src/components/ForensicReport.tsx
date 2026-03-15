import type { InvestigateResponse } from "../api/client";
import RiskBadge from "./RiskBadge";

interface Props {
  report: InvestigateResponse | null;
  loading?: boolean;
  error?: string | null;
}

export default function ForensicReport({ report, loading, error }: Props) {
  if (loading) {
    return (
      <div className="flex h-64 flex-col items-center justify-center gap-3 text-zinc-400">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-600 border-t-blue-400" />
        <p className="text-sm">Generating forensic report…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-800 bg-red-950/40 p-4 text-sm text-red-300">
        {error}
      </div>
    );
  }

  if (!report) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-zinc-600">
        Select a cluster and click "Investigate" to generate a forensic report.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <h3 className="text-lg font-semibold text-white">
          Cluster {report.cluster_id} Investigation
        </h3>
        <RiskBadge level={report.risk_level} />
        <span className="ml-auto text-sm text-zinc-400">
          Avg confidence: {(report.avg_confidence * 100).toFixed(1)}%
        </span>
      </div>

      {/* Markdown report rendered as pre-formatted text */}
      <div className="max-h-96 overflow-auto rounded-lg border border-zinc-800 bg-zinc-950 p-4">
        <pre className="whitespace-pre-wrap text-xs leading-relaxed text-zinc-300 font-mono">
          {report.report}
        </pre>
      </div>

      {report.similar_cases.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-zinc-300">Similar Historical Cases</h4>
          <div className="flex flex-col gap-2">
            {report.similar_cases.map((c, i) => (
              <div key={i} className="rounded border border-zinc-800 bg-zinc-900 p-3 text-xs text-zinc-400">
                <span className="font-semibold text-zinc-300">
                  Cluster {(c.metadata as { cluster_id?: number }).cluster_id ?? "?"} ·{" "}
                  {(c.metadata as { risk_level?: string }).risk_level ?? ""}
                </span>
                <p className="mt-1">{c.text_snippet}…</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
