from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

try:
    from dgl.nn.pytorch import GraphConv
    import dgl
except Exception:  # pragma: no cover - dgl is an optional runtime dependency.
    GraphConv = None
    dgl = None
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import to_dense_batch
except Exception:  # pragma: no cover
    GCNConv = None
    to_dense_batch = None


class DrugBAN(nn.Module):
    """DrugBAN architecture aligned with the official implementation.

    Official source used for structure:
    https://github.com/peizhenbai/DrugBAN/blob/main/models.py
    https://github.com/peizhenbai/DrugBAN/blob/main/ban.py

    The paper code uses DGL/DGLLife. This implementation keeps the same
    MolecularGCN -> ProteinCNN -> BANLayer -> MLPDecoder pipeline and replaces
    DGLLife's GCN wrapper with equivalent DGL ``GraphConv`` layers.
    """

    def __init__(
        self,
        node_in_feats: int = 78,
        node_embedding: int = 128,
        drug_hidden_feats: tuple[int, ...] = (128, 128, 128),
        protein_embedding_dim: int = 128,
        target_vocab_size: int = 26,
        protein_num_filters: tuple[int, int, int] = (128, 128, 128),
        protein_kernel_size: tuple[int, int, int] = (3, 6, 9),
        mlp_in_dim: int = 256,
        mlp_hidden_dim: int = 512,
        mlp_out_dim: int = 128,
        ban_heads: int = 2,
        out_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.drug_extractor = MolecularGCN(node_in_feats, node_embedding, drug_hidden_feats)
        self.protein_extractor = ProteinCNN(target_vocab_size, protein_embedding_dim, protein_num_filters, protein_kernel_size)
        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=protein_num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads, dropout=dropout),
            name="h_mat",
            dim=None,
        )
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, out_dim)

    def forward(self, graph, protein: torch.Tensor, mode: str = "score"):
        drug_features = self.drug_extractor(graph)
        protein_features = self.protein_extractor(protein)
        fusion, att = self.bcn(drug_features, protein_features)
        score = self.mlp_classifier(fusion)
        if mode == "train":
            return drug_features, protein_features, fusion, score
        if mode == "eval":
            return drug_features, protein_features, score, att
        return score.squeeze(-1)

    def forward_batch(self, data):
        graph = data.graph if hasattr(data, "graph") else data
        protein = data.target if hasattr(data, "target") else data.protein
        return self.forward(graph, protein)


class MolecularGCN(nn.Module):
    def __init__(self, in_feats: int, dim_embedding: int = 128, hidden_feats: tuple[int, ...] = (128, 128, 128)) -> None:
        super().__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        self.use_dgl = GraphConv is not None
        conv_cls = GraphConv if self.use_dgl else GCNConv
        if conv_cls is None:
            raise ImportError("DrugBAN requires either dgl or torch_geometric.")
        self.gnn = nn.ModuleList()
        last_dim = dim_embedding
        for hidden_dim in hidden_feats:
            if self.use_dgl:
                self.gnn.append(conv_cls(last_dim, hidden_dim, activation=F.relu, allow_zero_in_degree=True))
            else:
                self.gnn.append(conv_cls(last_dim, hidden_dim))
            last_dim = hidden_dim
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        if hasattr(batch_graph, "ndata"):
            return self._forward_dgl(batch_graph)
        return self._forward_pyg(batch_graph)

    def _forward_dgl(self, batch_graph):
        node_feats = batch_graph.ndata["h"].float()
        node_feats = self.init_transform(node_feats)
        for layer in self.gnn:
            node_feats = layer(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        return node_feats.view(batch_size, -1, self.output_feats)

    def _forward_pyg(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        node_feats = self.init_transform(x)
        for layer in self.gnn:
            node_feats = F.relu(layer(node_feats, edge_index))
        if to_dense_batch is not None:
            dense, _ = to_dense_batch(node_feats, batch)
            return dense
        batch_size = int(batch.max().item()) + 1
        max_nodes = int(torch.bincount(batch, minlength=batch_size).max().item())
        dense = node_feats.new_zeros(batch_size, max_nodes, self.output_feats)
        offsets = torch.zeros(batch_size, dtype=torch.long, device=batch.device)
        for node_idx, graph_id in enumerate(batch.tolist()):
            dense[graph_id, offsets[graph_id]] = node_feats[node_idx]
            offsets[graph_id] += 1
        return dense


class ProteinCNN(nn.Module):
    def __init__(self, target_vocab_size: int, embedding_dim: int, num_filters: tuple[int, int, int], kernel_size: tuple[int, int, int], padding: bool = True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(target_vocab_size, embedding_dim, padding_idx=0 if padding else None)
        in_ch = [embedding_dim] + list(num_filters)
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernel_size[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, protein: torch.Tensor) -> torch.Tensor:
        protein = self.embedding(protein.long()).transpose(2, 1)
        protein = self.bn1(F.relu(self.conv1(protein)))
        protein = self.bn2(F.relu(self.conv2(protein)))
        protein = self.bn3(F.relu(self.conv3(protein)))
        return protein.transpose(2, 1)


class BANLayer(nn.Module):
    def __init__(self, v_dim: int, q_dim: int, h_dim: int, h_out: int, act: str = "ReLU", dropout: float = 0.2, k: int = 3) -> None:
        super().__init__()
        self.c = 32
        self.k = k
        self.h_dim = h_dim
        self.h_out = h_out
        self.v_net = FCNet([v_dim, h_dim * k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * k], act=act, dropout=dropout)
        if 1 < k:
            self.p_net = nn.AvgPool1d(k, stride=k)
        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * k, h_out), dim=None)
        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v: torch.Tensor, q: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        fusion_logits = torch.einsum("bvk,bvq,bqk->bk", (v, att_map, q))
        if 1 < self.k:
            fusion_logits = self.p_net(fusion_logits.unsqueeze(1)).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v: torch.Tensor, q: torch.Tensor, softmax: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum("xhyk,bvk,bqk->bhvq", (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3)).transpose(2, 3).transpose(1, 2)
        if softmax:
            att_maps = F.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2).view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits = logits + self.attention_pooling(v_, q_, att_maps[:, i, :, :])
        return self.bn(logits), att_maps


class FCNet(nn.Module):
    def __init__(self, dims: list[int], act: str = "ReLU", dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(dims[i], dims[i + 1]), dim=None))
            if act:
                layers.append(getattr(nn, act)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class MLPDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, output: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        return self.fc4(x)
