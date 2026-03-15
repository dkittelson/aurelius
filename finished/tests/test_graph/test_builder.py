"""Tests for graph construction from CSV datasets.

These tests use synthetic data that mirrors the real dataset structure,
so they run without needing the actual Kaggle downloads.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures: synthetic CSVs that mimic real dataset structure
# ---------------------------------------------------------------------------

@pytest.fixture
def elliptic_raw_dir(tmp_path):
    """Create a temp directory with synthetic Elliptic CSVs."""
    num_nodes = 100
    num_edges = 150

    # features: col0=txId, col1=timestep, cols2-167=features
    node_ids = list(range(1, num_nodes + 1))
    timesteps = [((i % 49) + 1) for i in range(num_nodes)]  # 1-49
    features = np.random.rand(num_nodes, 166).astype(np.float32)

    feat_data = np.column_stack([node_ids, timesteps, features])
    pd.DataFrame(feat_data).to_csv(
        tmp_path / "elliptic_txs_features.csv", index=False, header=False
    )

    # edges
    src = np.random.choice(node_ids, num_edges)
    dst = np.random.choice(node_ids, num_edges)
    pd.DataFrame({"txId1": src, "txId2": dst}).to_csv(
        tmp_path / "elliptic_txs_edgelist.csv", index=False
    )

    # classes: 40 illicit, 40 licit, 20 unknown
    classes = (
        [(nid, "1") for nid in node_ids[:40]]
        + [(nid, "2") for nid in node_ids[40:80]]
        + [(nid, "unknown") for nid in node_ids[80:]]
    )
    pd.DataFrame(classes, columns=["txId", "class"]).to_csv(
        tmp_path / "elliptic_txs_classes.csv", index=False
    )

    return str(tmp_path)


@pytest.fixture
def ibm_raw_dir(tmp_path):
    """Create a temp directory with a synthetic IBM AML CSV."""
    num_rows = 200
    rng = np.random.default_rng(42)

    banks = ["BankA", "BankB", "BankC"]
    accounts = [f"ACC{i:03d}" for i in range(20)]

    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=num_rows, freq="h"),
        "From Bank": rng.choice(banks, num_rows),
        "Account": rng.choice(accounts, num_rows),
        "To Bank": rng.choice(banks, num_rows),
        "Account.1": rng.choice(accounts, num_rows),
        "Amount Received": rng.uniform(100, 10000, num_rows),
        "Receiving Currency": "USD",
        "Amount Paid": rng.uniform(100, 10000, num_rows),
        "Payment Currency": "USD",
        "Payment Format": rng.choice(["ACH", "Wire", "Check"], num_rows),
        "Is Laundering": rng.integers(0, 2, num_rows),
    })
    df.to_csv(tmp_path / "HI-Small_Trans.csv", index=False)
    return str(tmp_path)


@pytest.fixture
def config():
    """Minimal config matching config.yaml structure."""
    return {
        "data": {
            "elliptic": {
                "classes_file": "elliptic_txs_classes.csv",
                "features_file": "elliptic_txs_features.csv",
                "edgelist_file": "elliptic_txs_edgelist.csv",
                "num_features": 166,
                "num_classes": 3,
            },
            "ibm_aml": {
                "transactions_file": "HI-Small_Trans.csv",
            },
        },
        "graph": {
            "builder": {
                "self_loops": True,
                "undirected": False,
                "time_window_steps": 49,
            }
        },
    }


# ---------------------------------------------------------------------------
# EllipticGraphBuilder tests
# ---------------------------------------------------------------------------

class TestEllipticGraphBuilder:

    def test_graph_node_feature_shape(self, elliptic_raw_dir, config):
        """Node features should be [num_nodes, 166]."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph()
        assert data.x.shape[1] == 166
        assert data.x.shape[0] > 0

    def test_graph_edge_index_shape(self, elliptic_raw_dir, config):
        """edge_index should be [2, num_edges]."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph()
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0

    def test_labels_valid_values(self, elliptic_raw_dir, config):
        """Labels should only contain {-1, 0, 1}."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph()
        unique_labels = set(data.y.numpy().tolist())
        assert unique_labels.issubset({-1, 0, 1}), f"Unexpected labels: {unique_labels}"

    def test_illicit_and_licit_both_present(self, elliptic_raw_dir, config):
        """Both illicit (1) and licit (0) labels should be present."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph()
        assert (data.y == 1).sum() > 0, "No illicit nodes found"
        assert (data.y == 0).sum() > 0, "No licit nodes found"

    def test_temporal_masks_no_overlap(self, elliptic_raw_dir, config):
        """train, val, test masks must not overlap."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph()
        train = data.train_mask
        val = data.val_mask
        test = data.test_mask
        assert not (train & val).any(), "Train and val masks overlap"
        assert not (train & test).any(), "Train and test masks overlap"
        assert not (val & test).any(), "Val and test masks overlap"

    def test_temporal_masks_respect_timesteps(self, elliptic_raw_dir, config):
        """train_mask must only contain nodes from timesteps 1-34."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph()
        train_timesteps = data.timestep[data.train_mask]
        assert (train_timesteps >= 1).all() and (train_timesteps <= 34).all(), \
            "Train mask includes nodes outside timesteps 1-34"

    def test_no_data_leakage(self, elliptic_raw_dir, config):
        """Test mask should not contain any timesteps from train range."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph()
        test_timesteps = data.timestep[data.test_mask]
        if len(test_timesteps) > 0:
            assert (test_timesteps >= 43).all(), \
                "Test mask contains nodes from training timesteps (data leakage)"

    def test_edge_index_in_bounds(self, elliptic_raw_dir, config):
        """All edge indices must be valid node indices."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph()
        n = data.num_nodes
        assert data.edge_index.max() < n
        assert data.edge_index.min() >= 0

    def test_timestep_attribute_exists(self, elliptic_raw_dir, config):
        """data.timestep should be present with correct shape."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph()
        assert hasattr(data, "timestep")
        assert data.timestep.shape[0] == data.num_nodes

    def test_build_single_timestep(self, elliptic_raw_dir, config):
        """Building a single-timestep graph should return only nodes from that timestep."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph(timestep=1)
        assert (data.timestep == 1).all(), "Single-timestep graph has nodes from other timesteps"

    def test_save_and_load_graph(self, elliptic_raw_dir, config, tmp_path):
        """Saved and reloaded graph should match original."""
        from src.graph.builder import EllipticGraphBuilder
        builder = EllipticGraphBuilder(elliptic_raw_dir, config)
        data = builder.build_graph()
        save_path = str(tmp_path / "test_graph.pt")
        builder.save_graph(data, save_path)
        loaded = EllipticGraphBuilder.load_graph(save_path)
        assert loaded.num_nodes == data.num_nodes
        assert loaded.num_edges == data.num_edges
        assert torch.equal(loaded.y, data.y)


# ---------------------------------------------------------------------------
# IBMAMLGraphBuilder tests
# ---------------------------------------------------------------------------

class TestIBMAMLGraphBuilder:

    def test_heterogeneous_node_types(self, ibm_raw_dir, config):
        """HeteroData should have 'account' and 'bank' node types."""
        from src.graph.builder import IBMAMLGraphBuilder
        builder = IBMAMLGraphBuilder(ibm_raw_dir, config)
        data = builder.build_heterogeneous_graph()
        assert "account" in data.node_types
        assert "bank" in data.node_types

    def test_heterogeneous_edge_types(self, ibm_raw_dir, config):
        """HeteroData should have transaction and membership edge types."""
        from src.graph.builder import IBMAMLGraphBuilder
        builder = IBMAMLGraphBuilder(ibm_raw_dir, config)
        data = builder.build_heterogeneous_graph()
        edge_types = [str(et) for et in data.edge_types]
        assert any("transacts" in et for et in edge_types), \
            f"No 'transacts' edge type found. Got: {edge_types}"
        assert any("belongs_to" in et for et in edge_types), \
            f"No 'belongs_to' edge type found. Got: {edge_types}"

    def test_account_features_exist(self, ibm_raw_dir, config):
        """Account nodes must have feature matrix."""
        from src.graph.builder import IBMAMLGraphBuilder
        builder = IBMAMLGraphBuilder(ibm_raw_dir, config)
        data = builder.build_heterogeneous_graph()
        assert data["account"].x is not None
        assert data["account"].x.shape[0] > 0
        assert data["account"].x.shape[1] > 0

    def test_transaction_labels_binary(self, ibm_raw_dir, config):
        """Transaction labels should be 0 or 1."""
        from src.graph.builder import IBMAMLGraphBuilder
        builder = IBMAMLGraphBuilder(ibm_raw_dir, config)
        data = builder.build_heterogeneous_graph()
        labels = data["account", "transacts", "account"].y
        unique = set(labels.numpy().tolist())
        assert unique.issubset({0, 1}), f"Unexpected label values: {unique}"

    def test_transaction_edge_count(self, ibm_raw_dir, config):
        """Number of transaction edges should match number of CSV rows."""
        from src.graph.builder import IBMAMLGraphBuilder
        builder = IBMAMLGraphBuilder(ibm_raw_dir, config)
        data = builder.build_heterogeneous_graph()
        num_tx_edges = data["account", "transacts", "account"].edge_index.shape[1]
        assert num_tx_edges == 200  # matches fixture row count

    def test_belongs_to_edge_index_in_bounds(self, ibm_raw_dir, config):
        """belongs_to edges must reference valid account and bank indices."""
        from src.graph.builder import IBMAMLGraphBuilder
        builder = IBMAMLGraphBuilder(ibm_raw_dir, config)
        data = builder.build_heterogeneous_graph()
        ei = data["account", "belongs_to", "bank"].edge_index
        num_accounts = data["account"].x.shape[0]
        num_banks = data["bank"].x.shape[0]
        assert ei[0].max() < num_accounts
        assert ei[1].max() < num_banks
