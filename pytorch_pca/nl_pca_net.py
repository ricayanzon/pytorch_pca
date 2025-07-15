from typing import Optional

import torch


class NLPCANet:
    """Neural network for Non-Linear Principal Component Analysis.

    Args:
        units_per_layer: Number of units in each layer
        weight_decay: L2 regularization parameter
        data_dist: Distribution mask for missing data
        scaling_factor: Scaling factor for data normalization
    """

    def __init__(
        self,
        units_per_layer: list[int],
        weight_decay: float,
        data_dist: torch.Tensor,
        scaling_factor: float,
    ):
        if len(units_per_layer) < 2:
            raise ValueError("Network must have at least 2 layers")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")

        self.units_per_layer = units_per_layer
        self.weight_decay = weight_decay
        self.data_dist = data_dist
        self.scaling_factor = scaling_factor
        self.weights: Optional[torch.Tensor] = None

        self.layer_indices: list[tuple[int, int, int]] = []
        offset = 0
        for i in range(1, len(units_per_layer)):
            n_in = units_per_layer[i - 1]
            n_out = units_per_layer[i]
            w_size = n_in * n_out
            b_size = n_out
            self.layer_indices.append(
                (offset, offset + w_size, offset + w_size + b_size)
            )
            offset += w_size + b_size
