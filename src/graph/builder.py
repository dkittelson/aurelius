import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Data
import numpy as np


class EllipticGraphBuilder:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.classes_path = self.data_dir / "elliptic_txs_classes.csv"
        self.edges_path = self.data_dir / "elliptic_txs_edgelist.csv"
        self.features_path = self.data_dir / "elliptic_txs_features.csv"


    def build(self, save: bool = False):
        classes = pd.read_csv(self.classes_path) # (203769, 2)
        edges = pd.read_csv(self.edges_path) # (234355, 2)
        features = pd.read_csv(self.features_path) # (203768, 167)

        # --- Node ID Remapping --- 
        node_ids = features.iloc[:, 0].values 
        id_to_idx = {} 
        for i, nid in enumerate(node_ids): 
            id_to_idx[nid] = i # assings nid value with index (1: node_id, 2: node_id)

        # --- Edge Tensor --- 
        src_list = []
        dst_list = []
        src = edges.iloc[:, 0].map(id_to_idx) # convert source IDs to indices
        dst = edges.iloc[:, 1].map(id_to_idx) # convert destination IDs to indices
        mask = src.notna() & dst.notna() # keep only valid edges
        edge_index = torch.from_numpy(
            np.array([src[mask].astype(int).values, dst[mask].astype(int).values])
        ).long() # [2, 234000]

        # --- Feature Tensor ---
        x = torch.tensor(features.iloc[:, 2:].values, dtype=torch.float32)

        # --- Labels Mapping ---
        label_map = {"1": 1, "2": 0, "unknown": 2}
        id_to_label = {}
        for _, row in classes.iterrows():
            id_to_label[row["txId"]] = label_map[row["class"]] 
        
        # --- Label Tensor ---
        y_list = []
        for nid in node_ids:
            y_list.append(id_to_label.get(nid, 2))
        y = torch.tensor(y_list, dtype=torch.long)

        # --- Get Timestep ---
        timestep = torch.tensor(features.iloc[:, 1].values, dtype=torch.long)

        # --- Timeline Masks ---
        train_mask = (y != 2) & (timestep <= 34) & (timestep >= 1)
        val_mask = (y != 2) & (timestep >= 35) & (timestep <= 42)
        test_mask = (timestep >= 43) & (timestep <= 49)

        return Data(x=x, edge_index=edge_index, y=y, timestep=timestep, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)




                




        



        




