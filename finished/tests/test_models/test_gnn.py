"""Tests for AureliusGAT and AureliusGATHetero."""

import pytest
import torch
from torch_geometric.data import Data, HeteroData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_homogeneous_graph(num_nodes=30, in_channels=16, seed=42):
    """Small synthetic homogeneous graph for fast testing."""
    torch.manual_seed(seed)
    src = torch.randint(0, num_nodes, (60,))
    dst = torch.randint(0, num_nodes, (60,))
    # Add self-loops (as builder.py does)
    self_loops = torch.arange(num_nodes)
    edge_index = torch.stack([
        torch.cat([src, self_loops]),
        torch.cat([dst, self_loops]),
    ])
    x = torch.rand(num_nodes, in_channels)
    y = torch.randint(0, 2, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)


def make_heterogeneous_graph(num_accounts=20, num_banks=3, in_channels=8):
    """Small synthetic heterogeneous graph for HeteroGAT tests."""
    torch.manual_seed(0)
    data = HeteroData()
    data["account"].x = torch.rand(num_accounts, in_channels)
    data["bank"].x = torch.eye(num_banks)

    # account -> transacts -> account
    src = torch.randint(0, num_accounts, (40,))
    dst = torch.randint(0, num_accounts, (40,))
    data["account", "transacts", "account"].edge_index = torch.stack([src, dst])

    # account -> belongs_to -> bank
    acct_ids = torch.arange(num_accounts)
    bank_ids = torch.randint(0, num_banks, (num_accounts,))
    data["account", "belongs_to", "bank"].edge_index = torch.stack([acct_ids, bank_ids])

    return data


@pytest.fixture
def graph():
    return make_homogeneous_graph()


@pytest.fixture
def hetero_graph():
    return make_heterogeneous_graph()


# ---------------------------------------------------------------------------
# AureliusGAT — forward pass
# ---------------------------------------------------------------------------

class TestAureliusGATForward:

    def test_output_shape(self, graph):
        """logits should be [num_nodes, out_channels]."""
        from src.models.gnn_model import AureliusGAT
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                             num_heads=4, num_layers=3)
        out = model(graph.x, graph.edge_index)
        assert out.shape == (graph.num_nodes, 2)

    def test_output_dtype(self, graph):
        """Output should be float32."""
        from src.models.gnn_model import AureliusGAT
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2, num_heads=4)
        out = model(graph.x, graph.edge_index)
        assert out.dtype == torch.float32

    def test_single_layer(self, graph):
        """Works with num_layers=1."""
        from src.models.gnn_model import AureliusGAT
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                             num_heads=4, num_layers=1)
        out = model(graph.x, graph.edge_index)
        assert out.shape == (graph.num_nodes, 2)

    def test_no_residual(self, graph):
        """Works with residual=False."""
        from src.models.gnn_model import AureliusGAT
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                             num_heads=4, residual=False)
        out = model(graph.x, graph.edge_index)
        assert out.shape == (graph.num_nodes, 2)

    def test_different_num_layers(self, graph):
        """Output shape is consistent across different layer counts."""
        from src.models.gnn_model import AureliusGAT
        for n_layers in [1, 2, 3, 4]:
            model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                                 num_heads=4, num_layers=n_layers)
            out = model(graph.x, graph.edge_index)
            assert out.shape == (graph.num_nodes, 2), \
                f"Failed for num_layers={n_layers}"

    def test_jk_modes(self, graph):
        """All JK modes produce correct output shape."""
        from src.models.gnn_model import AureliusGAT
        for mode in ["cat", "max"]:
            model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                                 num_heads=4, num_layers=3, jk_mode=mode)
            out = model(graph.x, graph.edge_index)
            assert out.shape == (graph.num_nodes, 2), f"Failed for jk_mode={mode}"

    def test_hidden_channels_must_be_divisible_by_num_heads(self):
        """AssertionError when hidden_channels % num_heads != 0."""
        from src.models.gnn_model import AureliusGAT
        with pytest.raises(AssertionError):
            AureliusGAT(in_channels=16, hidden_channels=33, out_channels=2, num_heads=4)

    def test_no_nan_in_output(self, graph):
        """Forward pass should not produce NaN values."""
        from src.models.gnn_model import AureliusGAT
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2, num_heads=4)
        out = model(graph.x, graph.edge_index)
        assert not torch.isnan(out).any(), "NaN detected in model output"


# ---------------------------------------------------------------------------
# AureliusGAT — return_embeddings
# ---------------------------------------------------------------------------

class TestAureliusGATEmbeddings:

    def test_embedding_shape_cat_mode(self, graph):
        """With JK 'cat', embedding dim = num_layers * hidden_channels."""
        from src.models.gnn_model import AureliusGAT
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                             num_heads=4, num_layers=3, jk_mode="cat")
        logits, emb = model(graph.x, graph.edge_index, return_embeddings=True)
        expected_emb_dim = 3 * 32  # num_layers * hidden_channels
        assert emb.shape == (graph.num_nodes, expected_emb_dim)
        assert logits.shape == (graph.num_nodes, 2)

    def test_get_embeddings_matches_forward(self, graph):
        """get_embeddings() should return same shape as forward return_embeddings."""
        from src.models.gnn_model import AureliusGAT
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                             num_heads=4, num_layers=2, jk_mode="cat")
        model.eval()
        with torch.no_grad():
            _, emb_forward = model(graph.x, graph.edge_index, return_embeddings=True)
            emb_method = model.get_embeddings(graph.x, graph.edge_index)
        assert emb_forward.shape == emb_method.shape

    def test_embedding_dim_matches_jk_out_channels(self, graph):
        """jk_out_channels attribute should match actual embedding size."""
        from src.models.gnn_model import AureliusGAT
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                             num_heads=4, num_layers=3, jk_mode="cat")
        emb = model.get_embeddings(graph.x, graph.edge_index)
        assert emb.shape[1] == model.jk_out_channels


# ---------------------------------------------------------------------------
# AureliusGAT — return_attention
# ---------------------------------------------------------------------------

class TestAureliusGATAttention:

    def test_attention_returns_list(self, graph):
        """return_attention=True should return a list of tuples."""
        from src.models.gnn_model import AureliusGAT
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                             num_heads=4, num_layers=3)
        logits, attn_list = model(graph.x, graph.edge_index, return_attention=True)
        assert isinstance(attn_list, list)
        assert len(attn_list) == 3  # one per layer

    def test_attention_weights_shape(self, graph):
        """Attention weights should be [num_edges, num_heads]."""
        from src.models.gnn_model import AureliusGAT
        num_heads = 4
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                             num_heads=num_heads, num_layers=2)
        _, attn_list = model(graph.x, graph.edge_index, return_attention=True)
        for ei, alpha in attn_list:
            assert alpha.shape[1] == num_heads, \
                f"Expected {num_heads} heads, got {alpha.shape[1]}"
            assert alpha.shape[0] == ei.shape[1], \
                "Alpha should have one row per edge"

    def test_get_attention_weights_method(self, graph):
        """get_attention_weights() should return a list of per-layer tuples."""
        from src.models.gnn_model import AureliusGAT
        num_layers = 3
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                             num_heads=4, num_layers=num_layers)
        attn = model.get_attention_weights(graph.x, graph.edge_index)
        assert len(attn) == num_layers

    def test_both_flags_together(self, graph):
        """return_attention=True and return_embeddings=True together."""
        from src.models.gnn_model import AureliusGAT
        model = AureliusGAT(in_channels=16, hidden_channels=32, out_channels=2,
                             num_heads=4, num_layers=2)
        result = model(graph.x, graph.edge_index,
                       return_attention=True, return_embeddings=True)
        assert len(result) == 3  # (logits, embeddings, attention_list)
        logits, emb, attn = result
        assert logits.shape == (graph.num_nodes, 2)
        assert emb.shape[0] == graph.num_nodes
        assert isinstance(attn, list)


# ---------------------------------------------------------------------------
# AureliusGATHetero
# ---------------------------------------------------------------------------

class TestAureliusGATHetero:

    def test_forward_returns_dict(self, hetero_graph):
        """Forward should return a dict of {node_type: logits}."""
        from src.models.gnn_model import AureliusGATHetero
        metadata = hetero_graph.metadata()
        in_channels_dict = {
            "account": hetero_graph["account"].x.shape[1],
            "bank": hetero_graph["bank"].x.shape[1],
        }
        model = AureliusGATHetero(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=16,
            out_channels=2,
            num_heads=4,
            num_layers=2,
        )
        x_dict = {
            "account": hetero_graph["account"].x,
            "bank": hetero_graph["bank"].x,
        }
        edge_index_dict = {et: hetero_graph[et].edge_index for et in hetero_graph.edge_types}
        out = model(x_dict, edge_index_dict)
        assert isinstance(out, dict)

    def test_account_output_shape(self, hetero_graph):
        """Account node logits should be [num_accounts, out_channels]."""
        from src.models.gnn_model import AureliusGATHetero
        metadata = hetero_graph.metadata()
        in_channels_dict = {
            "account": hetero_graph["account"].x.shape[1],
            "bank": hetero_graph["bank"].x.shape[1],
        }
        model = AureliusGATHetero(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=16,
            out_channels=2,
            num_heads=4,
            num_layers=2,
        )
        x_dict = {
            "account": hetero_graph["account"].x,
            "bank": hetero_graph["bank"].x,
        }
        edge_index_dict = {et: hetero_graph[et].edge_index for et in hetero_graph.edge_types}
        out = model(x_dict, edge_index_dict)
        num_accounts = hetero_graph["account"].x.shape[0]
        assert out["account"].shape == (num_accounts, 2)
