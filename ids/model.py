import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from .data import knn_edges


class PayloadGraphIDS(nn.Module):
    def __init__(self, n_stats, n_classes, n_families, branch="both", graph=True):
        super().__init__()
        if branch not in ("both", "stats", "payload"):
            raise ValueError(branch)
        self.branch, self.graph = branch, graph
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(32, 32), nn.ReLU(),
        )
        self.stats_encoder = nn.Sequential(nn.Linear(n_stats, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        channels = 64 if branch == "both" else 32
        self.project = nn.Sequential(nn.Linear(channels + (1 if branch != "stats" else 0), 64), nn.ReLU())
        self.gat1 = GATv2Conv(64, 32, heads=2, concat=True, dropout=0.1)
        self.gat2 = GATv2Conv(64, 64, heads=1, concat=False)
        self.head = nn.Linear(64, n_classes)
        self.family_head = nn.Linear(64, n_families)

    def forward(self, stats, payload, mask):
        parts = []
        if self.branch != "payload":
            parts.append(self.stats_encoder(stats))
        if self.branch != "stats":
            parts.extend((self.cnn(payload.unsqueeze(1)) * mask, mask))
        embedding = self.project(torch.cat(parts, dim=1))
        if self.graph:
            edge_index = knn_edges(stats)
            embedding = embedding + torch.relu(self.gat2(torch.relu(self.gat1(embedding, edge_index)), edge_index))
        logits = self.head(embedding)
        return logits, self.family_head(embedding), embedding
