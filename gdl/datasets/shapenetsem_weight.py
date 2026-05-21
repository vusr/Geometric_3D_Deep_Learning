"""ShapeNetSem weight regression dataset with class and material side inputs."""

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .paths import resolve_manifest_path


class ShapeNetSemWeightDataset(Dataset):
    TARGET = "weight"

    def __init__(
        self,
        splits_csv: str | Path,
        stats_json: str | Path,
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

        with open(stats_json) as f:
            self.stats = json.load(f)
        self.target_stats = self.stats["target"]
        self.input_scale = float(self.stats.get("input_scale", 1.0))
        if not np.isfinite(self.input_scale) or self.input_scale <= 0:
            self.input_scale = 1.0

        self.aux_columns = list(self.stats["aux_columns"])
        self.class_to_idx = dict(self.stats["class_to_idx"])
        self.unknown_class_idx = int(self.class_to_idx.get("__unknown__", 0))

        self.manifest_dir = Path(splits_csv).expanduser().resolve().parent
        df = pd.read_csv(splits_csv)
        df = df[df["split"] == split].copy().reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No samples for split={split!r}")
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    @property
    def aux_dim(self) -> int:
        return len(self.aux_columns)

    @property
    def num_classes(self) -> int:
        return max(self.class_to_idx.values()) + 1

    def normalise_target(self, raw: float) -> np.ndarray:
        log_value = np.log1p(float(raw))
        mean = float(self.target_stats["log_mean"])
        std = max(float(self.target_stats["log_std"]), 1e-8)
        return np.asarray([(log_value - mean) / std], dtype=np.float32)

    def denormalise(self, norm_targets: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(float(self.target_stats["log_mean"]), dtype=norm_targets.dtype, device=norm_targets.device)
        std = torch.tensor(float(self.target_stats["log_std"]), dtype=norm_targets.dtype, device=norm_targets.device)
        return torch.expm1(norm_targets[..., 0] * std + mean).unsqueeze(-1)

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

        class_idx = self.class_to_idx.get(str(row["class_name"]), self.unknown_class_idx)
        aux = row[self.aux_columns].to_numpy(dtype=np.float32)
        raw_weight = float(row[self.TARGET])
        out = (
            torch.from_numpy(points.astype(np.float32)),
            torch.from_numpy(aux),
            torch.tensor(class_idx, dtype=torch.long),
            torch.from_numpy(self.normalise_target(raw_weight)),
        )
        if self.return_raw:
            return (*out, torch.tensor([raw_weight], dtype=torch.float32))
        return out
