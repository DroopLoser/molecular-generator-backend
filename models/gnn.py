#%%writefile models/gnn.py
"""
Graph Neural Network for multi-property molecular prediction.

Predicted properties:
    0. MolWt
    1. LogP
    2. TPSA
    3. HBD
    4. HBA
    5. QED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
)


class MolecularGNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_dim=128,
        out_dim=6,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Graph convolution layers
        self.conv1 = GCNConv(
            in_channels,
            hidden_dim,
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim,
        )

        # Output layer
        self.fc_out = nn.Linear(
            hidden_dim,
            out_dim,
        )

    def forward(
        self,
        x,
        edge_index,
        batch,
    ):
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Pool node embeddings into one graph embedding
        x = global_mean_pool(
            x,
            batch,
        )

        # Predict all properties
        out = self.fc_out(x)

        return out