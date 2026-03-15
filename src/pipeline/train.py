import torch
from torch_geometric.loader import NeighborLoader
from src.graph.builder import EllipticGraphBuilder
from src.models.gnn_model import AureliusGAT
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


class Trainer:
    def __init__(self, config: dict, device: str = "auto"):
        self.config = config
        self.device = torch.device(
            "mps" if device == "auto" and torch.backends.mps.is_available()
            else "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.data = None
        self.model = None
        self.train_loader = None

    def setup_data(self, dataset: str = "elliptic"):
        self.data = EllipticGraphBuilder('data/raw').build()
        self.train_loader = NeighborLoader(
            self.data,
            num_neighbors =[15,10,5],
            batch_size=2048,
            input_nodes = self.data.train_mask
        )

    def setup_model(self):
        self.model = AureliusGAT(in_channels=self.data.x.shape[1]).to(self.device)

    def train_gnn(self):
        cfg = self.config["model"]["training"]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"])

        # --- Class Weights ---
        num_licit = (self.data.y == 0).sum()
        num_ilicit = (self.data.y == 1).sum()
        weight = torch.tensor([1.0, num_licit / num_ilicit], dtype=torch.float32).to(self.device)

        # --- Training Loop ---
        best_auprc = 0
        patience_counter = 0
        for epoch in range(cfg["epochs"]):
            self.model.train()
            total_loss = 0

            # --- Training ---
            for batch in self.train_loader:
                batch = batch.to(self.device) 
                optimizer.zero_grad() # clear old gradients
                logits = self.model(batch.x, batch.edge_index) # forward pass
                loss = F.cross_entropy(logits[:batch.batch_size], 
                                       batch.y[:batch.batch_size], weight=weight) # compute loss
                loss.backward() # compute gradients (backprop)
                optimizer.step() # update weights
                total_loss += loss.item() 
            print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

            # --- Validation --- 
            self.model.eval()
            with torch.no_grad(): # don't track gradients
                all_data = self.data.to(self.device) 
                logits = self.model(all_data.x, all_data.edge_index)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            mask = self.data.train_mask.cpu().numpy()
            labels = self.data.y.cpu().numpy()
            auprc = average_precision_score(labels[mask], probs[mask])
            print(f"  Val AUPRC: {auprc:.4f}")

            if auprc > best_auprc:
                best_auprc = auprc
                patience_counter = 0
                torch.save(self.model.state_dict(), "data/processed/checkpoints/best_gnn.pt")
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    print("Early stopping.")
                    break

