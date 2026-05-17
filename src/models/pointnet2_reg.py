"""
PointNet++ MSG Regression Model.

Architecture identical to the classification model except the output head
produces 2 continuous values: [normalised_log_volume, normalised_log_area].

Architecture:
  Input  : batch of (B*N, 3) point clouds + batch vector
  SA1    : MSG at ratio=0.5, radii=[0.1, 0.2, 0.4]  → 320-dim features
  SA2    : MSG at ratio=0.25, radii=[0.2, 0.4, 0.8] → 640-dim features
  Global : global max-pool + MLP → 1024-dim global descriptor
  Head   : FC(1024→512) → FC(512→256) → FC(256→2)
"""

import torch
import torch.nn as nn
from torch import Tensor

from .pointnet2_utils import MsgSAModule, GlobalSAModule, build_mlp


class PointNet2Regression(nn.Module):
    """
    Args:
        out_dim  : Number of regression outputs (2 = volume + area)
        dropout  : Dropout rate applied in regression head
    """

    def __init__(self, out_dim: int = 2, dropout: float = 0.3):
        super().__init__()

        # SA1 — same MSG config as classification model
        self.sa1 = MsgSAModule(
            ratio=0.5,
            radii=[0.1, 0.2, 0.4],
            nns=[
                build_mlp([3, 32, 32, 64]),
                build_mlp([3, 64, 64, 128]),
                build_mlp([3, 64, 96, 128]),
            ],
        )

        # SA2
        self.sa2 = MsgSAModule(
            ratio=0.25,
            radii=[0.2, 0.4, 0.8],
            nns=[
                build_mlp([320 + 3, 64, 64, 128]),
                build_mlp([320 + 3, 128, 128, 256]),
                build_mlp([320 + 3, 128, 128, 256]),
            ],
        )

        # Global SA
        self.sa3 = GlobalSAModule(
            nn=build_mlp([640 + 3, 256, 512, 1024])
        )

        # Regression head — no activation on final layer (unbounded outputs)
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, out_dim),  # no activation — predict normalised targets
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pos: Tensor, batch: Tensor) -> Tensor:
        """
        Args:
            pos   : (B*N, 3) — point coordinates
            batch : (B*N,)   — batch index per point
        Returns:
            preds : (B, 2) — normalised [log_volume, log_area] predictions
        """
        x, pos, batch = self.sa1(None, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, pos, batch = self.sa3(x, pos, batch)  # (B, 1024)
        return self.head(x)
