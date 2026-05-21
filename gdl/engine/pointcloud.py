"""Point-cloud batching helpers shared by experiment entrypoints."""

import torch


def collate_label_batch(batch):
    points, labels = zip(*batch)
    return torch.stack(points, dim=0), torch.stack(labels, dim=0)


def collate_target_batch(batch):
    points, targets = zip(*batch)
    return torch.stack(points, dim=0), torch.stack(targets, dim=0)


def collate_aux_batch(batch):
    points, aux, class_idx, targets = zip(*batch)
    return (
        torch.stack(points, dim=0),
        torch.stack(aux, dim=0),
        torch.stack(class_idx, dim=0),
        torch.stack(targets, dim=0),
    )


def points_to_pyg(points: torch.Tensor, device):
    batch_size, n_points, _ = points.shape
    pos = points.reshape(batch_size * n_points, 3).to(device)
    batch = torch.arange(batch_size, device=device).repeat_interleave(n_points)
    return pos, batch
