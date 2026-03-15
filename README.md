


**Datasets**
Elliptic Dataset: 
- elliptic_txs_features.csv: Each row is a bitcoin transaction (node)
- - Transaction ID
- - Timestep
- - 166 numeric features (how much BTC sent, how many inputs/outputs, etc)

**Phase 1: Data Preprocessing & Graph Construction**
builder.py:
1. Reads CSVs
2. Assigns each transaction index (0,1,2,..)
3. Creates tensor for features of every transaction node x = [203k nodes, 166 features]
4. Creates edge_index tensor of shape [2, num_edges] -- "node A connects to node B"
5. Creates labels y -- 0 for licit, 1 for illicit, -1 for unknown
6. Creates temporal masks --> timesteps 1-34 (train), 35-42 (val), 42-49 (test)

features.py:
- Adds 5 features based on the shape of the network
1. PageRank: How important is a node based on its connections --> hub accounts used for layering have high PageRank
2. Degree centrality: How many direct connections --> smurfing (many small deposits into one account)
3. Betweenness centrality: How often a node sits between others --> layer accounts are on path between dirty source and clean destination
4. Closeness centrality: How quickly a node can reach all others --> highly connected laundering hubs reach all others quickly
5. Local clustering coefficient: Are a node's neighbors connected to eachother? --> legit businesses cluster

**Phase 2: GNN Model & Training**
gnn_model.py:
- Attention Heads: Each layer runs 4 independent "attention heads" --> each learns different way to weigh node's neighbors
- Jumping Knowledge: Concatenate outputs from all 3 layers (layer 1: 1-hop neighbors; layer 2: 2-hop neighbors) --> local and long-range signals
- Residual Connections: Each layer, add input back to the output --> prevents gradients from vanishing 

classifier.py:
- Takes 384-dim GNN embedding per node, concatenates it with og 166+ features, giving a 550-dim tabular row
- XGBoost learns feature splits GNN can't learn (sharp dollar thresholds, exchange rate ratios)

train.py:
- Each training step loads 2048 nodes, then samples up to 15 neighbors per seed at hop 1, 10 at hop 2, 5 at hop 3
- early stops after 20 epochs if no improvement







