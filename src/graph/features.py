import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

class GraphFeatureEngineer:
    def __init__(self, graph: Data):
        self.graph = graph

    def compute_all(self) -> Data:
        G = to_networkx(self.graph, to_undirected=True)
        pagerank = nx.pagerank(G, alpha=0.85)
        degree     = nx.degree_centrality(G)
        between = nx.betweenness_centrality(G, k=500)
        closeness  = nx.closeness_centrality(G)
        clustering = nx.clustering(G)

        # create feature tensor
        n = self.graph.num_nodes
        pr = [pagerank[i] for i in range(n)]
        deg = [degree[i] for i in range(n)]
        bet = [between[i] for i in range(n)]
        clo = [closeness[i] for i in range(n)]
        clu = [clustering[i] for i in range(n)]
        features = torch.tensor([pr, deg, bet, clo, clu], dtype=torch.float32).T

        # concatenate onto graph
        self.graph.x = torch.cat([self.graph.x, features], dim=1)

        return self.graph
