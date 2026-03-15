"""Graph construction from CSV datasets to PyTorch Geometric objects."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from torch_geometric.data import Data, HeteroData
from loguru import logger

from src.graph.download import find_file


class EllipticGraphBuilder:
    """Builds PyG Data from Elliptic Bitcoin CSV files.

    The Elliptic dataset contains ~203K Bitcoin transactions across 49 timesteps.
    Each transaction has 166 features (94 local + 72 aggregated neighbor features).
    Labels: 1=illicit, 2=licit, unknown=unlabeled.
    """

    def __init__(self, raw_dir: str, config: dict):
        self.raw_dir = raw_dir
        self.config = config

    def load_csvs(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load features, edges, and classes CSVs."""
        elliptic_cfg = self.config["data"]["elliptic"]

        features_path = find_file(elliptic_cfg["features_file"], self.raw_dir)
        edges_path = find_file(elliptic_cfg["edgelist_file"], self.raw_dir)
        classes_path = find_file(elliptic_cfg["classes_file"], self.raw_dir)

        logger.info(f"Loading Elliptic features from {features_path}")
        features_df = pd.read_csv(features_path, header=None)

        logger.info(f"Loading Elliptic edges from {edges_path}")
        edges_df = pd.read_csv(edges_path)

        logger.info(f"Loading Elliptic classes from {classes_path}")
        classes_df = pd.read_csv(classes_path)

        return features_df, edges_df, classes_df

    def build_graph(self, timestep: Optional[int] = None) -> Data:
        """
        Construct a PyG Data object from the Elliptic dataset.

        Args:
            timestep: If provided, only include nodes from this timestep (1-49).
                      If None, builds the full graph across all timesteps.

        Returns:
            Data object with:
                - x: [num_nodes, 166] node features
                - edge_index: [2, num_edges] directed edges
                - y: [num_nodes] labels (0=licit, 1=illicit, -1=unknown)
                - train_mask, val_mask, test_mask: temporal split masks
                - timestep: [num_nodes] timestep assignment
                - node_ids: [num_nodes] original transaction IDs
        """
        features_df, edges_df, classes_df = self.load_csvs()

        # Column 0 = transaction ID, column 1 = timestep, columns 2-167 = features
        node_ids = features_df.iloc[:, 0].values
        timesteps = features_df.iloc[:, 1].values.astype(int)
        features = features_df.iloc[:, 2:].values.astype(np.float32)

        # Map original IDs to contiguous indices
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Parse labels: "1"=illicit, "2"=licit, "unknown"=-1
        classes_df.columns = ["txId", "class"]
        label_map = {}
        for _, row in classes_df.iterrows():
            txid = row["txId"]
            cls = row["class"]
            if cls == "unknown" or cls == "3":
                label_map[txid] = -1
            elif str(cls).strip() == "1":
                label_map[txid] = 1  # illicit
            else:
                label_map[txid] = 0  # licit

        labels = np.array([label_map.get(nid, -1) for nid in node_ids], dtype=np.int64)

        # Filter by timestep if specified
        if timestep is not None:
            mask = timesteps == timestep
            node_ids = node_ids[mask]
            timesteps = timesteps[mask]
            features = features[mask]
            labels = labels[mask]
            id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Build edge index
        src_col = edges_df.columns[0]
        dst_col = edges_df.columns[1]
        valid_ids = set(node_ids)

        edges_src, edges_dst = [], []
        for _, row in edges_df.iterrows():
            s, d = row[src_col], row[dst_col]
            if s in valid_ids and d in valid_ids:
                edges_src.append(id_to_idx[s])
                edges_dst.append(id_to_idx[d])

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        # Add self-loops if configured
        if self.config["graph"]["builder"].get("self_loops", True):
            n = len(node_ids)
            self_loops = torch.arange(n, dtype=torch.long).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)

        # Temporal split: train=1-34, val=35-42, test=43-49
        train_mask = torch.tensor(
            [bool(1 <= t <= 34) and bool(labels[i] != -1) for i, t in enumerate(timesteps)],
            dtype=torch.bool,
        )
        val_mask = torch.tensor(
            [bool(35 <= t <= 42) and bool(labels[i] != -1) for i, t in enumerate(timesteps)],
            dtype=torch.bool,
        )
        test_mask = torch.tensor(
            [bool(43 <= t <= 49) and bool(labels[i] != -1) for i, t in enumerate(timesteps)],
            dtype=torch.bool,
        )

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        ts = torch.tensor(timesteps, dtype=torch.int32)
        nids = torch.tensor(node_ids, dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            timestep=ts,
            node_ids=nids,
        )

        logger.info(
            f"Built Elliptic graph: {data.num_nodes} nodes, "
            f"{data.num_edges} edges, "
            f"illicit={int((y == 1).sum())}, licit={int((y == 0).sum())}, "
            f"unknown={int((y == -1).sum())}"
        )

        return data

    def build_temporal_snapshots(self) -> list[Data]:
        """Build one graph per timestep (49 snapshots)."""
        snapshots = []
        for t in range(1, 50):
            try:
                snap = self.build_graph(timestep=t)
                snapshots.append(snap)
            except Exception as e:
                logger.warning(f"Failed to build snapshot for timestep {t}: {e}")
        return snapshots

    def save_graph(self, data: Data, path: str = "data/processed/elliptic_graph.pt"):
        """Save processed graph to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_path)
        logger.info(f"Saved graph to {save_path}")

    @staticmethod
    def load_graph(path: str = "data/processed/elliptic_graph.pt") -> Data:
        """Load processed graph from disk."""
        return torch.load(Path(path), weights_only=False)


class IBMAMLGraphBuilder:
    """Builds PyG HeteroData from IBM Synthetic AML CSV.

    The IBM AML dataset contains synthetic banking transactions with
    money laundering labels on transactions.
    """

    def __init__(self, raw_dir: str, config: dict):
        self.raw_dir = raw_dir
        self.config = config

    def load_csv(self) -> pd.DataFrame:
        """Load and parse HI-Small_Trans.csv."""
        ibm_cfg = self.config["data"]["ibm_aml"]
        filepath = find_file(ibm_cfg["transactions_file"], self.raw_dir)

        logger.info(f"Loading IBM AML transactions from {filepath}")
        df = pd.read_csv(filepath)

        # Parse timestamp if present
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        return df

    def build_heterogeneous_graph(self) -> HeteroData:
        """
        Construct HeteroData with:
        - Node types: 'account', 'bank'
        - Edge types: ('account','transacts','account'), ('account','belongs_to','bank')
        - Node features: aggregated transaction statistics per account
        - Edge labels: Is Laundering (0/1) on transaction edges
        """
        df = self.load_csv()

        # Identify columns (IBM AML format)
        from_bank_col = "From Bank"
        from_acct_col = "Account"
        to_bank_col = "To Bank"
        to_acct_col = "Account.1"
        amount_col = "Amount Received"
        label_col = "Is Laundering"

        # Adjust column names if they differ
        col_map = {c: c for c in df.columns}
        for c in df.columns:
            cl = c.lower().strip()
            if "from" in cl and "bank" in cl:
                from_bank_col = c
            elif "to" in cl and "bank" in cl:
                to_bank_col = c
            elif "amount" in cl and "received" in cl:
                amount_col = c
            elif "laundering" in cl:
                label_col = c

        # Create unique account IDs by combining bank + account
        df["from_id"] = df[from_bank_col].astype(str) + "_" + df[from_acct_col].astype(str)
        df["to_id"] = df[to_bank_col].astype(str) + "_" + df[to_acct_col].astype(str)

        # Build account index
        all_accounts = sorted(set(df["from_id"].unique()) | set(df["to_id"].unique()))
        acct_to_idx = {a: i for i, a in enumerate(all_accounts)}
        num_accounts = len(all_accounts)

        # Build bank index
        all_banks = sorted(
            set(df[from_bank_col].astype(str).unique())
            | set(df[to_bank_col].astype(str).unique())
        )
        bank_to_idx = {b: i for i, b in enumerate(all_banks)}
        num_banks = len(all_banks)

        # Account features: aggregate transaction statistics (vectorized)
        account_features = self._build_account_features(df, acct_to_idx, num_accounts, amount_col)

        # Bank features: simple one-hot (identity)
        bank_features = torch.eye(num_banks, dtype=torch.float32)

        # Transaction edges: vectorized map instead of list comprehension
        src_indices = df["from_id"].map(acct_to_idx).values.astype(np.int64)
        dst_indices = df["to_id"].map(acct_to_idx).values.astype(np.int64)
        tx_edge_index = torch.tensor(np.stack([src_indices, dst_indices]), dtype=torch.long)

        # Transaction labels
        tx_labels = torch.tensor(df[label_col].values.astype(np.int64), dtype=torch.long)

        # Transaction edge features (amount)
        amounts = df[amount_col].fillna(0).values.astype(np.float32)
        tx_edge_attr = torch.tensor(amounts, dtype=torch.float32).unsqueeze(1)

        # Account -> Bank edges (vectorized)
        acct_ids_series = pd.Series(list(acct_to_idx.keys()), name="acct_id")
        acct_idx_series = pd.Series(list(acct_to_idx.values()), name="acct_idx")
        bank_names = acct_ids_series.str.split("_").str[0]
        valid_mask = bank_names.isin(bank_to_idx)
        acct_bank_src = acct_idx_series[valid_mask].values.tolist()
        acct_bank_dst = bank_names[valid_mask].map(bank_to_idx).values.tolist()
        belongs_to_edge_index = torch.tensor(
            [acct_bank_src, acct_bank_dst], dtype=torch.long
        )

        # Build HeteroData
        data = HeteroData()
        data["account"].x = account_features
        data["bank"].x = bank_features
        data["account", "transacts", "account"].edge_index = tx_edge_index
        data["account", "transacts", "account"].edge_attr = tx_edge_attr
        data["account", "transacts", "account"].y = tx_labels
        data["account", "belongs_to", "bank"].edge_index = belongs_to_edge_index

        logger.info(
            f"Built IBM AML graph: {num_accounts} accounts, {num_banks} banks, "
            f"{len(df)} transactions, "
            f"laundering={int(tx_labels.sum())}/{len(tx_labels)}"
        )

        return data

    def _build_account_features(
        self, df: pd.DataFrame, acct_to_idx: dict, num_accounts: int, amount_col: str
    ) -> torch.Tensor:
        """Aggregate per-account statistics from transaction DataFrame (fully vectorized)."""
        # Sender aggregations (groupby from_id)
        sent_grp = df.groupby("from_id")[amount_col].agg(["sum", "count", "mean"]).fillna(0)
        sent_grp.columns = ["total_sent", "tx_count_out", "avg_sent"]

        # Receiver aggregations (groupby to_id)
        recv_grp = df.groupby("to_id")[amount_col].agg(["sum", "count", "mean"]).fillna(0)
        recv_grp.columns = ["total_received", "tx_count_in", "avg_received"]

        # Unique counterparties per account (both directions combined)
        cp_out = df.groupby("from_id")["to_id"].nunique().rename("cp_out")
        cp_in = df.groupby("to_id")["from_id"].nunique().rename("cp_in")

        # Build master account DataFrame indexed by account ID
        all_acct_ids = list(acct_to_idx.keys())
        master = pd.DataFrame(index=all_acct_ids)
        master = master.join(sent_grp, how="left").fillna(0)
        master = master.join(recv_grp, how="left").fillna(0)
        master = master.join(cp_out, how="left").fillna(0)
        master = master.join(cp_in, how="left").fillna(0)

        master["tx_count_total"] = master["tx_count_out"] + master["tx_count_in"]
        master["n_counterparties"] = master["cp_out"] + master["cp_in"]

        # Reorder by acct_to_idx so row i = account index i
        master["__idx"] = master.index.map(acct_to_idx)
        master = master.sort_values("__idx")

        feature_cols = [
            "total_sent", "total_received", "tx_count_out", "tx_count_in",
            "tx_count_total", "avg_sent", "avg_received", "n_counterparties",
        ]

        features = master[feature_cols].values.astype(np.float32)

        # Normalize features (z-score)
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        features = (features - mean) / std

        return torch.tensor(features, dtype=torch.float32)

    def save_graph(self, data: HeteroData, path: str = "data/processed/ibm_aml_graph.pt"):
        """Save processed graph to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_path)
        logger.info(f"Saved graph to {save_path}")

    @staticmethod
    def load_graph(path: str = "data/processed/ibm_aml_graph.pt") -> HeteroData:
        """Load processed graph from disk."""
        return torch.load(Path(path), weights_only=False)
