"""
PointNet++ MSG Classification Model.

Architecture:
  Input  : batch of (B*N, 3) point clouds + batch vector
  SA1    : MSG at ratio=0.5, radii=[0.1, 0.2, 0.4]  → 320-dim features
  SA2    : MSG at ratio=0.25, radii=[0.2, 0.4, 0.8] → 640-dim features
  Global : global max-pool + MLP → 1024-dim global descriptor
  Head   : FC(1024→512) → FC(512→256) → FC(256→40 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pointnet2_utils import MsgSAModule, GlobalSAModule, build_mlp


class PointNet2Classification(nn.Module):
    """
    Args:
        num_classes : Number of output classes (40 for ModelNet40)
        dropout     : Dropout rate applied in classification head
    """

    def __init__(self, num_classes: int = 40, dropout: float = 0.4):
        super().__init__()

        # SA1 — MSG, 3 scales
        # Input features: None (raw XYZ only), so PointNetConv sees (pos_j - pos_i || 0)
        # Each MLP input = 3 (relative position) + 0 (no input features) = 3
        self.sa1 = MsgSAModule(
            ratio=0.5,
            radii=[0.1, 0.2, 0.4],
            nns=[
                build_mlp([3, 32, 32, 64]),    # scale 1 → 64
                build_mlp([3, 64, 64, 128]),   # scale 2 → 128
                build_mlp([3, 64, 96, 128]),   # scale 3 → 128
            ],
        )
        # SA1 output: 64 + 128 + 128 = 320

        # SA2 — MSG, 3 scales
        # Input features dim = 320 from SA1, so each MLP sees 3 + 320 = 323
        self.sa2 = MsgSAModule(
            ratio=0.25,
            radii=[0.2, 0.4, 0.8],
            nns=[
                build_mlp([320 + 3, 64, 64, 128]),    # scale 1 → 128
                build_mlp([320 + 3, 128, 128, 256]),  # scale 2 → 256
                build_mlp([320 + 3, 128, 128, 256]),  # scale 3 → 256
            ],
        )
        # SA2 output: 128 + 256 + 256 = 640

        # Global SA
        # Input: 640 + 3 = 643
        self.sa3 = GlobalSAModule(
            nn=build_mlp([640 + 3, 256, 512, 1024])
        )
        # Global SA output: 1024

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
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
            pos   : (B*N, 3) — point coordinates, all batches concatenated
            batch : (B*N,)   — batch index per point (0..B-1)
        Returns:
            logits : (B, num_classes)
        """
        # SA layers (no input features at first level — x=None)
        x, pos, batch = self.sa1(None, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, pos, batch = self.sa3(x, pos, batch)  # (B, 1024)

        return self.head(x)
