from __future__ import annotations

import math

import torch


def rmse_loss(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean((predictions - targets) ** 2) + eps)


def mean_euclidean_distance(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(predictions - targets, dim=1).mean()


def normdist(predictions: torch.Tensor, targets: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    diagonal = math.sqrt(image_width**2 + image_height**2)
    return mean_euclidean_distance(predictions, targets) / diagonal
