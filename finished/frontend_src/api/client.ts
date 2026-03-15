import axios from "axios";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";

export const api = axios.create({ baseURL: BASE });

// ── Types ────────────────────────────────────────────────────────────────────

export interface GraphStats {
  num_nodes: number;
  num_edges: number;
  num_illicit: number;
  num_licit: number;
  num_unknown: number;
  illicit_rate: number;
}

export interface NodeInfo {
  node_id: number;
  label: number;
  illicit_prob?: number;
  is_seed?: boolean;  // true = original cluster node, false = expanded neighbor
}

export interface SubgraphEdge {
  src: number;
  dst: number;
}

export interface SubgraphData {
  nodes: NodeInfo[];
  edges: SubgraphEdge[];
}

export interface NodePrediction {
  node_id: number;
  illicit_prob: number;
  risk_level: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
}

export interface TopKResponse {
  nodes: NodePrediction[];
  total_above_threshold: number;
}

export interface ClusterSummary {
  cluster_id: number;
  node_ids: number[];
  num_nodes: number;
  num_edges: number;
  avg_confidence: number;
  max_confidence: number;
  risk_level: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  density: number;
}

export interface ClustersResponse {
  clusters: ClusterSummary[];
  total_suspicious_nodes: number;
}

export interface InvestigateResponse {
  cluster_id: number;
  report: string;
  risk_level: string;
  avg_confidence: number;
  similar_cases: { metadata: Record<string, unknown>; text_snippet: string }[];
}

export interface DashboardStats {
  total_nodes: number;
  total_edges: number;
  flagged_nodes: number;
  critical_clusters: number;
  high_clusters: number;
  medium_clusters: number;
  low_clusters: number;
  model_val_auprc: number | null;
  dataset: string;
}

// ── API calls ────────────────────────────────────────────────────────────────

export const fetchGraphStats = () =>
  api.get<GraphStats>("/graph/stats").then((r) => r.data);

export const fetchSubgraph = (nodeIds: number[], expandHops = 1) =>
  api
    .post<SubgraphData>("/graph/subgraph", { node_ids: nodeIds, include_edges: true, expand_hops: expandHops })
    .then((r) => r.data);

export const fetchTopK = (k = 50, threshold = 0.5) =>
  api
    .post<TopKResponse>("/predictions/top-k", { k, threshold })
    .then((r) => r.data);

export const fetchClusters = (threshold = 0.75, minSize = 2, maxClusters = 10) =>
  api
    .post<ClustersResponse>("/forensic/clusters", {
      threshold,
      min_cluster_size: minSize,
      max_clusters: maxClusters,
    })
    .then((r) => r.data);

export const investigateCluster = (clusterId: number, nodeIds: number[], dataset = "elliptic") =>
  api
    .post<InvestigateResponse>("/forensic/investigate", {
      cluster_id: clusterId,
      node_ids: nodeIds,
      dataset,
    })
    .then((r) => r.data);

export const fetchDashboardStats = () =>
  api.get<DashboardStats>("/dashboard/stats").then((r) => r.data);

// ── Explainability ──────────────────────────────────────────────────────────

export interface ImportantEdge {
  src: number;
  dst: number;
  importance: number;
}

export interface ImportantFeature {
  feature_index: number;
  importance: number;
}

export interface NodeExplanation {
  node_id: number;
  important_edges: ImportantEdge[];
  important_features: ImportantFeature[];
  summary: string;
}

export interface ClusterExplanation {
  node_ids: number[];
  important_edges: ImportantEdge[];
  summary: string;
}

export const explainNode = (nodeId: number, topKEdges = 10, topKFeatures = 10) =>
  api
    .post<NodeExplanation>("/explain/node", {
      node_id: nodeId,
      top_k_edges: topKEdges,
      top_k_features: topKFeatures,
    })
    .then((r) => r.data);

export const explainCluster = (nodeIds: number[], topKEdges = 20) =>
  api
    .post<ClusterExplanation>("/explain/cluster", {
      node_ids: nodeIds,
      top_k_edges: topKEdges,
    })
    .then((r) => r.data);
