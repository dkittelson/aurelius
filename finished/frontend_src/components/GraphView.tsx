import { useRef, useState, useEffect } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { NodeInfo, SubgraphEdge } from "../api/client";

interface Props {
  nodes: NodeInfo[];
  edges: SubgraphEdge[];
  onNodeClick?: (nodeId: number) => void;
}

type FGNode = { id: number; illicit_prob: number; label: number; is_seed?: boolean };

function probToColor(prob: number): string {
  if (prob >= 0.9)  return "#ef4444";
  if (prob >= 0.75) return "#f97316";
  if (prob >= 0.6)  return "#eab308";
  return "#22c55e";
}

export default function GraphView({ nodes, edges, onNodeClick }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ width: 600, height: 400 });

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setDims({ width, height });
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  if (!nodes || nodes.length === 0) {
    return (
      <div className="flex h-full w-full items-center justify-center rounded-xl border border-zinc-800 bg-zinc-950 text-sm text-zinc-500">
        No graph data
      </div>
    );
  }

  const graphData = {
    nodes: nodes.map((n) => ({
      id: n.node_id,
      illicit_prob: n.illicit_prob ?? 0,
      label: n.label,
      is_seed: n.is_seed ?? false,
    })),
    // Filter self-loops — ForceGraph2D crashes on source === target
    links: (edges ?? [])
      .filter((e) => e.src !== e.dst)
      .map((e) => ({ source: e.src, target: e.dst })),
  };

  return (
    <div ref={containerRef} className="h-full w-full rounded-xl overflow-hidden bg-zinc-950 border border-zinc-800 relative">
      <ForceGraph2D
        graphData={graphData}
        width={dims.width}
        height={dims.height}
        nodeId="id"
        // Seed nodes (original cluster) are larger than expanded neighbors
        nodeRelSize={5}
        nodeVal={(n) => (n as FGNode).is_seed ? 3 : 1}
        nodeColor={(n) => probToColor((n as FGNode).illicit_prob)}
        // Seed nodes get a white ring to make them stand out
        nodeCanvasObjectMode={(n) => (n as FGNode).is_seed ? "before" : undefined}
        nodeCanvasObject={(n, ctx) => {
          const fn = n as FGNode & { x?: number; y?: number };
          if (!fn.is_seed || fn.x == null || fn.y == null) return;
          ctx.beginPath();
          ctx.arc(fn.x, fn.y, 9, 0, 2 * Math.PI);
          ctx.strokeStyle = "rgba(255,255,255,0.6)";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }}
        linkColor={() => "#3f3f46"}
        linkWidth={0.8}
        backgroundColor="#09090b"
        onNodeClick={(n) => onNodeClick?.((n as FGNode).id)}
        nodeLabel={(n) => {
          const fn = n as FGNode;
          const tag = fn.is_seed ? " [CLUSTER]" : "";
          return `Node ${fn.id}${tag} | p=${fn.illicit_prob.toFixed(3)}`;
        }}
      />
      <div className="pointer-events-none absolute bottom-3 left-3 flex gap-2 text-xs">
        {[
          { color: "#ef4444", label: "CRITICAL ≥0.9" },
          { color: "#f97316", label: "HIGH ≥0.75" },
          { color: "#eab308", label: "MEDIUM ≥0.6" },
          { color: "#22c55e", label: "LOW" },
        ].map(({ color, label }) => (
          <span key={label} className="flex items-center gap-1 rounded bg-zinc-900/80 px-2 py-0.5">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
