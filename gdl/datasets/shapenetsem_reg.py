"""
Dataset for ShapeNetSem geometry-core regression.

Targets are stored in raw metric units and normalized as train-only log1p
z-scores. Point clouds are centered but keep metric scale.
"""

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .paths import resolve_manifest_path


class ShapeNetSemRegDataset(Dataset):
    TARGETS = (
        "solidVolume",
        "surfaceVolume",
        "supportSurfaceArea",
        "aligned_dim_x",
        "aligned_dim_y",
        "aligned_dim_z",
    )

    def __init__(
        self,
        splits_csv: str | Path,
        norm_stats_json: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        n_points: Optional[int] = 1024,
        return_raw: bool = False,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split = split
        self.transform = transform
        self.n_points = n_points
        self.return_raw = return_raw

        with open(norm_stats_json) as f:
            self.stats = json.load(f)
        self.target_stats = self.stats["global"]
        self.input_scale = float(self.target_stats.get("input_scale", 1.0))
        if not np.isfinite(self.input_scale) or self.input_scale <= 0:
            self.input_scale = 1.0

        self.manifest_dir = Path(splits_csv).expanduser().resolve().parent
        df = pd.read_csv(splits_csv)
        df = df[df["split"] == split].copy().reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No samples for split={split!r}")
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def normalise(self, raw_values: np.ndarray) -> np.ndarray:
        values = []
        for name, raw in zip(self.TARGETS, raw_values):
            log_value = np.log1p(float(raw))
            mean = float(self.target_stats[name]["log_mean"])
            std = max(float(self.target_stats[name]["log_std"]), 1e-8)
            values.append((log_value - mean) / std)
        return np.asarray(values, dtype=np.float32)

    def denormalise(self, norm_targets: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, name in enumerate(self.TARGETS):
            stat = self.target_stats[name]
            mean = torch.tensor(float(stat["log_mean"]), dtype=norm_targets.dtype, device=norm_targets.device)
            std = torch.tensor(float(stat["log_std"]), dtype=norm_targets.dtype, device=norm_targets.device)
            outputs.append(torch.expm1(norm_targets[..., i] * std + mean))
        return torch.stack(outputs, dim=-1)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        data = np.load(resolve_manifest_path(row["npz_path"], self.manifest_dir))
        points = data["points_centered"].astype(np.float32)

        if self.n_points is not None and self.n_points < len(points):
            choice = np.random.choice(len(points), self.n_points, replace=False)
            points = points[choice]

        points = points / self.input_scale
        if self.transform is not None:
            points = self.transform(points)

        raw_values = row[list(self.TARGETS)].to_numpy(dtype=np.float32)
        points_tensor = torch.from_numpy(points.astype(np.float32))
        targets_normed = torch.from_numpy(self.normalise(raw_values))

        if self.return_raw:
            return points_tensor, targets_normed, torch.from_numpy(raw_values)
        return points_tensor, targets_normed

    @classmethod
    def target_names(cls):
        return list(cls.TARGETS)
