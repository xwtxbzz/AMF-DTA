from __future__ import annotations

import torch
from torch import nn
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GATConv, GCNConv, GINConv
from torch_geometric.nn import global_add_pool as gap_add
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class GraphDTA(nn.Module):
    """PyG implementation aligned with the official GraphDTA models.

    Official source used for structure:
    https://github.com/thinng/GraphDTA/tree/master/models

    ``variant`` chooses one of the paper/code variants: ``gcn``, ``gat``,
    ``gat_gcn`` or ``gin``. The forward interface accepts either a PyG Batch
    via ``forward_batch(data)`` or explicit ``x, edge_index, batch, target``.
    """

    def __init__(
        self,
        num_node_features: int = 78,
        target_vocab_size: int = 25,
        variant: str = "gcn",
        n_output: int = 1,
        n_filters: int = 32,
        embed_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.2,
        target_max_len: int = 1000,
    ) -> None:
        super().__init__()
        self.variant = variant.lower().replace("-", "_")
        if self.variant == "gcn":
            self.net = GCNNet(n_output, n_filters, embed_dim, num_node_features, target_vocab_size, output_dim, dropout, target_max_len)
        elif self.variant == "gat":
            self.net = GATNet(n_output, n_filters, embed_dim, num_node_features, target_vocab_size, output_dim, dropout, target_max_len)
        elif self.variant == "gat_gcn":
            self.net = GATGCNNet(n_output, n_filters, embed_dim, num_node_features, target_vocab_size, output_dim, dropout, target_max_len)
        elif self.variant == "gin":
            self.net = GINConvNet(n_output, n_filters, embed_dim, num_node_features, target_vocab_size, output_dim, dropout, target_max_len)
        else:
            raise ValueError("variant must be one of: gcn, gat, gat_gcn, gin")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.net(x.float(), edge_index, batch, target.long()).squeeze(-1)

    def forward_batch(self, data) -> torch.Tensor:
        target = data.target if hasattr(data, "target") else data.protein
        return self.forward(data.x, data.edge_index, data.batch, target)


class ProteinBranch(nn.Module):
    def __init__(self, target_vocab_size: int, embed_dim: int, n_filters: int, output_dim: int, target_max_len: int) -> None:
        super().__init__()
        self.embedding_xt = nn.Embedding(target_vocab_size + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=target_max_len, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(n_filters * (embed_dim - 8 + 1), output_dim)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        return self.fc1_xt(conv_xt.flatten(start_dim=1))


class GCNNet(nn.Module):
    def __init__(self, n_output, n_filters, embed_dim, num_features_xd, num_features_xt, output_dim, dropout, target_max_len) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = Linear(num_features_xd * 4, 1024)
        self.fc_g2 = Linear(1024, output_dim)
        self.protein = ProteinBranch(num_features_xt, embed_dim, n_filters, output_dim, target_max_len)
        self.fc1 = Linear(2 * output_dim, 1024)
        self.fc2 = Linear(1024, 512)
        self.out = Linear(512, n_output)
        self.relu = ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, target):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = gmp(x, batch)
        x = self.dropout(self.relu(self.fc_g1(x)))
        x = self.dropout(self.fc_g2(x))
        xt = self.protein(target)
        xc = torch.cat((x, xt), 1)
        xc = self.dropout(self.relu(self.fc1(xc)))
        xc = self.dropout(self.relu(self.fc2(xc)))
        return self.out(xc)


class GATNet(nn.Module):
    def __init__(self, n_output, n_filters, embed_dim, num_features_xd, num_features_xt, output_dim, dropout, target_max_len) -> None:
        super().__init__()
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = Linear(output_dim, output_dim)
        self.protein = ProteinBranch(num_features_xt, embed_dim, n_filters, output_dim, target_max_len)
        self.fc1 = Linear(2 * output_dim, 1024)
        self.fc2 = Linear(1024, 512)
        self.out = Linear(512, n_output)
        self.relu = ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, target):
        x = self.dropout(self.relu(self.gcn1(x, edge_index)))
        x = self.dropout(self.relu(self.gcn2(x, edge_index)))
        x = gmp(x, batch)
        x = self.relu(self.fc_g1(x))
        xt = self.protein(target)
        xc = torch.cat((x, xt), 1)
        xc = self.dropout(self.relu(self.fc1(xc)))
        xc = self.dropout(self.relu(self.fc2(xc)))
        return self.out(xc)


class GATGCNNet(nn.Module):
    def __init__(self, n_output, n_filters, embed_dim, num_features_xd, num_features_xt, output_dim, dropout, target_max_len) -> None:
        super().__init__()
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.fc_g1 = Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g2 = Linear(1500, output_dim)
        self.protein = ProteinBranch(num_features_xt, embed_dim, n_filters, output_dim, target_max_len)
        self.fc1 = Linear(2 * output_dim, 1024)
        self.fc2 = Linear(1024, 512)
        self.out = Linear(512, n_output)
        self.relu = ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, target):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.dropout(self.relu(self.fc_g1(x)))
        x = self.fc_g2(x)
        xt = self.protein(target)
        xc = torch.cat((x, xt), 1)
        xc = self.dropout(self.relu(self.fc1(xc)))
        xc = self.dropout(self.relu(self.fc2(xc)))
        return self.out(xc)


class GINConvNet(nn.Module):
    def __init__(self, n_output, n_filters, embed_dim, num_features_xd, num_features_xt, output_dim, dropout, target_max_len) -> None:
        super().__init__()
        nn1 = Sequential(Linear(num_features_xd, 32), ReLU(), Linear(32, 32))
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(32)
        nn2 = Sequential(Linear(32, 32), ReLU(), Linear(32, 32))
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(32)
        nn3 = Sequential(Linear(32, 32), ReLU(), Linear(32, 32))
        self.conv3 = GINConv(nn3)
        self.bn3 = nn.BatchNorm1d(32)
        nn4 = Sequential(Linear(32, 32), ReLU(), Linear(32, 32))
        self.conv4 = GINConv(nn4)
        self.bn4 = nn.BatchNorm1d(32)
        nn5 = Sequential(Linear(32, 32), ReLU(), Linear(32, 32))
        self.conv5 = GINConv(nn5)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc1_xd = Linear(32, output_dim)
        self.protein = ProteinBranch(num_features_xt, embed_dim, n_filters, output_dim, target_max_len)
        self.fc1 = Linear(2 * output_dim, 1024)
        self.fc2 = Linear(1024, 256)
        self.out = Linear(256, n_output)
        self.relu = ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, target):
        x = self.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = self.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = self.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = gap_add(x, batch)
        x = self.dropout(self.relu(self.fc1_xd(x)))
        xt = self.protein(target)
        xc = torch.cat((x, xt), 1)
        xc = self.dropout(self.relu(self.fc1(xc)))
        xc = self.dropout(self.relu(self.fc2(xc)))
        return self.out(xc)
