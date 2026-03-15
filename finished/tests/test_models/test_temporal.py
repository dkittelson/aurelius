"""Tests for Temporal GNN components."""

import pytest
import torch
from torch_geometric.data import Data

from src.models.temporal_gnn import TemporalNodeMemory, TemporalAureliusGAT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(num_nodes=20, in_channels=16, seed=42):
    torch.manual_seed(seed)
    src = torch.randint(0, num_nodes, (40,))
    dst = torch.randint(0, num_nodes, (40,))
    self_loops = torch.arange(num_nodes)
    edge_index = torch.stack([
        torch.cat([src, self_loops]),
        torch.cat([dst, self_loops]),
    ])
    x = torch.rand(num_nodes, in_channels)
    y = torch.randint(0, 2, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def snapshot():
    return _make_snapshot()


@pytest.fixture
def snapshot_sequence():
    return [_make_snapshot(seed=i) for i in range(3)]


# ---------------------------------------------------------------------------
# TemporalNodeMemory
# ---------------------------------------------------------------------------

class TestTemporalNodeMemory:
    def test_initial_memory_is_zero(self):
        mem = TemporalNodeMemory(embedding_dim=32, memory_dim=16, max_nodes=50)
        ids = torch.tensor([0, 5, 10])
        m = mem.get_memory(ids)
        assert m.shape == (3, 16)
        assert (m == 0).all()

    def test_update_returns_correct_shape(self):
        mem = TemporalNodeMemory(embedding_dim=32, memory_dim=16, max_nodes=50)
        emb = torch.randn(5, 32)
        ids = torch.tensor([0, 1, 2, 3, 4])
        result = mem.update(emb, ids)
        assert result.shape == (5, 16)

    def test_memory_persists_after_update(self):
        mem = TemporalNodeMemory(embedding_dim=32, memory_dim=16, max_nodes=50)
        emb = torch.randn(3, 32)
        ids = torch.tensor([0, 1, 2])
        mem.update(emb, ids)
        m = mem.get_memory(ids)
        assert not (m == 0).all()

    def test_reset_zeros_memory(self):
        mem = TemporalNodeMemory(embedding_dim=32, memory_dim=16, max_nodes=50)
        emb = torch.randn(3, 32)
        ids = torch.tensor([0, 1, 2])
        mem.update(emb, ids)
        mem.reset()
        m = mem.get_memory(ids)
        assert (m == 0).all()

    def test_update_only_affects_specified_nodes(self):
        mem = TemporalNodeMemory(embedding_dim=32, memory_dim=16, max_nodes=50)
        emb = torch.randn(2, 32)
        ids = torch.tensor([5, 10])
        mem.update(emb, ids)
        # Other nodes should still be zero
        other = mem.get_memory(torch.tensor([0, 1, 2]))
        assert (other == 0).all()


# ---------------------------------------------------------------------------
# TemporalAureliusGAT
# ---------------------------------------------------------------------------

class TestTemporalAureliusGAT:
    def test_forward_snapshot_output_shape(self, snapshot):
        model = TemporalAureliusGAT(
            in_channels=16, hidden_channels=32, memory_dim=16,
            out_channels=2, num_heads=4, num_layers=2,
            dropout=0.0, jk_mode="cat", max_nodes=100,
        )
        logits = model.forward_snapshot(snapshot)
        assert logits.shape == (snapshot.num_nodes, 2)

    def test_forward_snapshot_with_global_ids(self, snapshot):
        model = TemporalAureliusGAT(
            in_channels=16, hidden_channels=32, memory_dim=16,
            out_channels=2, num_heads=4, num_layers=2,
            dropout=0.0, jk_mode="cat", max_nodes=100,
        )
        global_ids = torch.arange(50, 50 + snapshot.num_nodes)
        logits = model.forward_snapshot(snapshot, global_ids)
        assert logits.shape == (snapshot.num_nodes, 2)

    def test_forward_sequence_returns_list(self, snapshot_sequence):
        model = TemporalAureliusGAT(
            in_channels=16, hidden_channels=32, memory_dim=16,
            out_channels=2, num_heads=4, num_layers=2,
            dropout=0.0, jk_mode="cat", max_nodes=100,
        )
        logits_list = model.forward_sequence(snapshot_sequence)
        assert len(logits_list) == 3
        for logits in logits_list:
            assert logits.shape[1] == 2

    def test_memory_accumulates_across_snapshots(self, snapshot_sequence):
        model = TemporalAureliusGAT(
            in_channels=16, hidden_channels=32, memory_dim=16,
            out_channels=2, num_heads=4, num_layers=2,
            dropout=0.0, jk_mode="cat", max_nodes=100,
        )
        model.memory.reset()
        model.forward_snapshot(snapshot_sequence[0])
        mem_after_1 = model.memory.memory[:20].clone()

        model.forward_snapshot(snapshot_sequence[1])
        mem_after_2 = model.memory.memory[:20].clone()

        # Memory should change after processing second snapshot
        assert not torch.allclose(mem_after_1, mem_after_2)

    def test_gradient_flows(self, snapshot):
        model = TemporalAureliusGAT(
            in_channels=16, hidden_channels=32, memory_dim=16,
            out_channels=2, num_heads=4, num_layers=2,
            dropout=0.0, jk_mode="cat", max_nodes=100,
        )
        model.train()
        logits = model.forward_snapshot(snapshot)
        loss = logits.sum()
        loss.backward()
        # Check that encoder parameters received gradients
        for p in model.encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_reset_memory_flag_in_sequence(self, snapshot_sequence):
        model = TemporalAureliusGAT(
            in_channels=16, hidden_channels=32, memory_dim=16,
            out_channels=2, num_heads=4, num_layers=2,
            dropout=0.0, jk_mode="cat", max_nodes=100,
        )
        # First run
        model.forward_sequence(snapshot_sequence, reset_memory=True)
        mem1 = model.memory.memory[:20].clone()

        # Second run with reset
        model.forward_sequence(snapshot_sequence, reset_memory=True)
        mem2 = model.memory.memory[:20].clone()

        # Should be the same since we reset and used same inputs
        assert torch.allclose(mem1, mem2, atol=1e-5)
