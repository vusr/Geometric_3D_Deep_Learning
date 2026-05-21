"""
Classification dataset for ModelNet40.

Loads pre-processed .npz files (produced by tools/data/prepare_modelnet40_classification_data.py) and
returns (points [N,3], class_id) tensors ready for PointNet++.

Optional augmentations are applied only when split='train'.
"""

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CLASS_NAMES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]
NUM_CLASSES = len(CLASS_NAMES)


class ModelNet40ClsDataset(Dataset):
    """
    Args:
        splits_csv  : Path to data/splits.csv
        split       : One of 'train', 'val', 'test'
        transform   : Optional callable applied to the (N,3) numpy array
                      before converting to tensor (e.g. RandomRotation)
        n_points    : Number of points to randomly sub-sample at load time
                      (set to None to keep the full 1024)
    """

    def __init__(
        self,
        splits_csv: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        n_points: Optional[int] = 1024,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split = split
        self.transform = transform
        self.n_points = n_points

        df = pd.read_csv(splits_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split='{split}' in {splits_csv}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        data = np.load(row["npz_path"])
        points = data["points"].astype(np.float32)  # (1024, 3)

        # Optional random sub-sampling (useful for data augmentation / ablation)
        if self.n_points is not None and self.n_points < len(points):
            choice = np.random.choice(len(points), self.n_points, replace=False)
            points = points[choice]

        if self.transform is not None:
            points = self.transform(points)

        points_tensor = torch.from_numpy(points)           # (N, 3)
        label_tensor = torch.tensor(int(row["class_id"]), dtype=torch.long)
        return points_tensor, label_tensor

    @property
    def class_names(self):
        return CLASS_NAMES

    @property
    def num_classes(self):
        return NUM_CLASSES

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for optional weighted loss."""
        counts = self.df["class_id"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(np.float32)
        weights = weights / weights.sum()
        return torch.from_numpy(weights)
