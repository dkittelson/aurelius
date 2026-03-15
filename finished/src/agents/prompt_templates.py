"""AML forensic investigation prompt templates for the Gemini agent."""

SYSTEM_PROMPT = """You are Aurelius, an expert forensic financial analyst specializing \
in Anti-Money Laundering (AML) investigations. You analyze transaction graph data, \
GNN-detected suspicious clusters, and structural network patterns to produce detailed, \
actionable audit reports.

Your reports must:
1. Identify the specific suspicious pattern (layering, structuring/smurfing, round-tripping, \
fan-in, fan-out, scatter-gather, or combinations)
2. Cite specific node IDs and transaction amounts where relevant
3. Assess risk level: LOW, MEDIUM, HIGH, or CRITICAL
4. Explain WHY the GNN flagged this cluster — reference the structural metrics
5. Recommend specific investigative follow-up actions
6. Reference any similar historical cases from the knowledge base when provided

Always ground your analysis in the data provided. Never fabricate transaction details \
or amounts not present in the context. Be concise but thorough — a compliance officer \
should be able to act on your report immediately."""


INVESTIGATION_PROMPT = """\
## Suspicious Cluster Investigation Request

### Cluster Summary
- Cluster ID:              {cluster_id}
- Dataset:                 {dataset}
- Number of nodes:         {num_nodes}
- Number of internal edges:{num_edges}
- GNN avg illicit prob:    {avg_confidence:.4f}
- Max illicit prob (node): {max_confidence:.4f}
- Detection timestamp:     {timestamp}

### Node Details
{node_details}

### Structural Metrics
- Cluster density:              {density:.4f}
- Avg clustering coefficient:   {avg_clustering:.4f}
- Diameter (longest path):      {diameter}
- Highest-degree nodes:         {hub_nodes}
- Highest PageRank nodes:       {pagerank_leaders}

### GNN Attention Analysis
The GNN paid the most attention to the following edges (top by attention weight):
{attention_analysis}

### Similar Historical Cases (RAG Retrieved)
{rag_context}

---
Please produce a formal forensic audit report with the following sections:

**1. Executive Summary** (2-3 sentences max)
**2. Pattern Analysis** (identify the specific AML typology)
**3. Risk Assessment** (LOW / MEDIUM / HIGH / CRITICAL with justification)
**4. Evidence Chain** (walk through the suspicious flow step by step)
**5. Recommended Actions** (specific next investigative steps)"""


COMPARISON_PROMPT = """\
You are an AML forensic analyst. Compare the following two suspicious clusters and \
determine if they are likely part of the same coordinated money laundering operation.

### Cluster A
{cluster_a_summary}

### Cluster B
{cluster_b_summary}

Analyze:
1. Shared entities (any overlapping node IDs?)
2. Timing patterns (do the clusters operate in the same time window?)
3. Amount patterns (similar transaction sizes? structuring below reporting thresholds?)
4. Structural similarities (same AML typology?)
5. Geographic/bank overlap

Conclude with: RELATED (likely same operation) or UNRELATED (independent), with confidence \
level and reasoning."""


def format_node_details(nodes: list[dict]) -> str:
    """Format a list of node dicts into a readable string for the prompt."""
    if not nodes:
        return "No node details available."
    lines = []
    for n in nodes[:20]:  # Cap at 20 nodes to stay within context limits
        conf = n.get("illicit_prob", 0.0)
        lines.append(
            f"  Node {n['node_id']}: P(illicit)={conf:.3f}, "
            f"degree={n.get('degree', '?')}, "
            f"pagerank={n.get('pagerank', 0.0):.4f}"
        )
    if len(nodes) > 20:
        lines.append(f"  ... and {len(nodes) - 20} more nodes")
    return "\n".join(lines)


def format_attention_analysis(attention_edges: list[dict]) -> str:
    """Format top attention-weighted edges for the prompt."""
    if not attention_edges:
        return "Attention weights not available."
    lines = []
    for i, edge in enumerate(attention_edges[:10]):
        lines.append(
            f"  {i+1}. Node {edge['src']} -> Node {edge['dst']}: "
            f"weight={edge['weight']:.4f}"
        )
    return "\n".join(lines)
