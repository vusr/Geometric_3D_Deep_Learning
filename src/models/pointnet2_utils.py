"""
PointNet++ building blocks using PyTorch Geometric primitives.

Key components:
  - SAModule       : Single-scale Set Abstraction
  - MsgSAModule    : Multi-Scale Grouping Set Abstraction
  - GlobalSAModule : Global pooling (no sampling / grouping radius)

References:
  Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets
  in a Metric Space", NeurIPS 2017.
"""

from typing import List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius


# ---------------------------------------------------------------------------
# Single-scale Set Abstraction
# ---------------------------------------------------------------------------

class SAModule(nn.Module):
    """
    Set Abstraction with a single radius.

    Args:
        ratio    : FPS sub-sampling ratio (fraction of input points kept)
        r        : Ball query radius
        nn       : MLP applied to each (x_j - x_i || h_j) pair
                   Input dim should be 3 + in_channels.
    """

    def __init__(self, ratio: float, r: float, nn: nn.Module):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(
        self,
        x: Optional[Tensor],
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Optional[Tensor], Tensor, Tensor]:
        """
        Returns:
            x_out   : (M, out_channels) — features at new centroids
            pos_out : (M, 3)            — centroid positions
            batch_out: (M,)             — batch vector
        """
        # Farthest point sampling → centroid indices
        idx = fps(pos, batch, ratio=self.ratio)
        pos_i = pos[idx]
        batch_i = batch[idx]

        # Ball query: for each centroid find neighbours within radius r
        row, col = radius(pos, pos_i, self.r, batch, batch_i, max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)  # [source, target]

        x_out = self.conv(x, (pos, pos_i), edge_index)
        return x_out, pos_i, batch_i


# ---------------------------------------------------------------------------
# Multi-Scale Grouping Set Abstraction
# ---------------------------------------------------------------------------

class MsgSAModule(nn.Module):
    """
    Multi-Scale Grouping Set Abstraction.

    Applies multiple SAModules at different radii, then concatenates results.

    Args:
        ratio       : FPS ratio (shared across scales)
        radii       : List of ball-query radii [r1, r2, ...]
        n_samples   : Max neighbours per radius (ignored by radius(), kept for docs)
        nns         : List of MLP modules — one per radius scale
    """

    def __init__(
        self,
        ratio: float,
        radii: List[float],
        nns: List[nn.Module],
    ):
        super().__init__()
        assert len(radii) == len(nns), "radii and nns must have equal length"
        self.ratio = ratio
        self.radii = radii
        self.convs = nn.ModuleList(
            [PointNetConv(nn_i, add_self_loops=False) for nn_i in nns]
        )

    def forward(
        self,
        x: Optional[Tensor],
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Shared FPS — centroids are the same for all scales
        idx = fps(pos, batch, ratio=self.ratio)
        pos_i = pos[idx]
        batch_i = batch[idx]

        scale_features = []
        for r, conv in zip(self.radii, self.convs):
            row, col = radius(pos, pos_i, r, batch, batch_i, max_num_neighbors=64)
            edge_index = torch.stack([col, row], dim=0)
            feat = conv(x, (pos, pos_i), edge_index)  # (M, out_c)
            scale_features.append(feat)

        x_out = torch.cat(scale_features, dim=-1)  # (M, sum(out_c))
        return x_out, pos_i, batch_i


# ---------------------------------------------------------------------------
# Global Set Abstraction
# ---------------------------------------------------------------------------

class GlobalSAModule(nn.Module):
    """
    Global max-pooling over all remaining points.

    Args:
        nn : MLP applied per point before pooling.
               Input dim should be 3 + in_channels.
    """

    def __init__(self, nn: nn.Module):
        super().__init__()
        self.nn = nn

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Concatenate position and features, apply MLP, then global max-pool
        x_in = torch.cat([x, pos], dim=-1) if x is not None else pos
        x_out = self.nn(x_in)
        x_out = global_max_pool(x_out, batch)   # (B, out_channels)

        # Dummy pos / batch for compatibility with downstream modules
        pos_out = pos.new_zeros((x_out.size(0), 3))
        batch_out = torch.arange(x_out.size(0), device=batch.device)
        return x_out, pos_out, batch_out


# ---------------------------------------------------------------------------
# MLP helper (wrapper around PyG's MLP)
# ---------------------------------------------------------------------------

def build_mlp(channel_list: List[int], dropout: float = 0.0, batch_norm: bool = True) -> nn.Module:
    """Build a PyG MLP with optional BatchNorm and dropout."""
    return MLP(
        channel_list,
        dropout=dropout,
        batch_norm=batch_norm,
        act="relu",
    )
