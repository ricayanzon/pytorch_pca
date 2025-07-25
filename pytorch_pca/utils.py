from typing import Tuple

import torch


def _check_data(data: torch.Tensor) -> bool:
    """Check if data format is valid."""
    if not isinstance(data, torch.Tensor):
        raise TypeError("Data must be a PyTorch tensor.")
    if data.dim() != 2:
        raise ValueError("Data must be a 2D tensor (samples x features).")
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("Data must have at least 2 samples and 2 features.")
    if torch.isnan(data).all():
        raise ValueError("Data cannot be all NaN values.")
    if (~torch.isnan(data)).sum().item() < 2:
        raise ValueError("Data must have at least 2 valid (non-NaN) values.")
    return True


def _center_data(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Center data by subtracting the mean."""
    means = torch.nanmean(data, dim=0, keepdim=True)
    centered_data = data - means
    return centered_data, means


def _scale_data(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale data to unit variance."""
    valid_mask = ~torch.isnan(data)
    means = torch.sum(data * valid_mask, dim=0) / torch.sum(valid_mask, dim=0)
    variance = torch.sum(((data - means) ** 2) * valid_mask, dim=0) / (
        torch.sum(valid_mask, dim=0) - 1
    )
    stds = torch.sqrt(variance).unsqueeze(0)
    stds = torch.where(stds == 0, torch.ones_like(stds), stds)
    scaled_data = data / stds
    return scaled_data, stds


def normalize_data(
    data: torch.Tensor, center: bool = True, scale: bool = False
) -> torch.Tensor:
    """Normalize data by centering and/or scaling."""
    result = data.clone()
    if center:
        result, _ = _center_data(result)
    if scale:
        result, _ = _scale_data(result)
    return result
