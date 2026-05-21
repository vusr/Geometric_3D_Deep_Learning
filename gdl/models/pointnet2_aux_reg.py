"""PointNet++ MSG regression with class and material auxiliary inputs."""

import torch
import torch.nn as nn
from torch import Tensor

from .pointnet2_utils import GlobalSAModule, MsgSAModule, build_mlp


class PointNet2AuxRegression(nn.Module):
    """Predict continuous targets from a point cloud plus tabular side inputs."""

    def __init__(
        self,
        out_dim: int = 1,
        num_classes: int = 1,
        aux_dim: int = 0,
        class_embed_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.sa1 = MsgSAModule(
            ratio=0.5,
            radii=[0.1, 0.2, 0.4],
            nns=[
                build_mlp([3, 32, 32, 64]),
                build_mlp([3, 64, 64, 128]),
                build_mlp([3, 64, 96, 128]),
            ],
        )
        self.sa2 = MsgSAModule(
            ratio=0.25,
            radii=[0.2, 0.4, 0.8],
            nns=[
                build_mlp([320 + 3, 64, 64, 128]),
                build_mlp([320 + 3, 128, 128, 256]),
                build_mlp([320 + 3, 128, 128, 256]),
            ],
        )
        self.sa3 = GlobalSAModule(nn=build_mlp([640 + 3, 256, 512, 1024]))
        self.class_embedding = nn.Embedding(max(int(num_classes), 1), class_embed_dim)
        head_in = 1024 + class_embed_dim + int(aux_dim)
        self.head = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pos: Tensor, batch: Tensor, aux: Tensor, class_idx: Tensor) -> Tensor:
        x, pos, batch = self.sa1(None, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, _, _ = self.sa3(x, pos, batch)
        class_features = self.class_embedding(class_idx.long())
        if aux.numel() == 0:
            merged = torch.cat([x, class_features], dim=1)
        else:
            merged = torch.cat([x, class_features, aux], dim=1)
        return self.head(merged)
