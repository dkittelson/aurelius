import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchGraphStats,
  fetchTopK,
  fetchClusters,
  fetchSubgraph,
  fetchDashboardStats,
  investigateCluster,
  explainNode,
} from "../api/client";

export const useGraphStats = () =>
  useQuery({ queryKey: ["graphStats"], queryFn: fetchGraphStats, staleTime: 60_000 });

export const useDashboardStats = () =>
  useQuery({ queryKey: ["dashboardStats"], queryFn: fetchDashboardStats, staleTime: 30_000 });

export const useTopK = (k = 50, threshold = 0.5) =>
  useQuery({
    queryKey: ["topK", k, threshold],
    queryFn: () => fetchTopK(k, threshold),
    staleTime: 60_000,
  });

export const useClusters = (threshold = 0.75, minSize = 2, maxClusters = 10) =>
  useQuery({
    queryKey: ["clusters", threshold, minSize, maxClusters],
    queryFn: () => fetchClusters(threshold, minSize, maxClusters),
    staleTime: 60_000,
  });

export const useSubgraph = (nodeIds: number[], expandHops = 2) =>
  useQuery({
    queryKey: ["subgraph", nodeIds, expandHops],
    queryFn: () => fetchSubgraph(nodeIds, expandHops),
    enabled: nodeIds.length > 0,
    staleTime: 300_000,
  });

export const useInvestigate = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      clusterId,
      nodeIds,
      dataset,
    }: {
      clusterId: number;
      nodeIds: number[];
      dataset?: string;
    }) => investigateCluster(clusterId, nodeIds, dataset),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["clusters"] }),
  });
};

export const useExplainNode = () =>
  useMutation({
    mutationFn: ({ nodeId, topKEdges }: { nodeId: number; topKEdges?: number }) =>
      explainNode(nodeId, topKEdges),
  });
