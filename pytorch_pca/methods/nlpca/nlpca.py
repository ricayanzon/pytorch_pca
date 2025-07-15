from typing import Optional

import torch

from ...pca_result import PCAResult
from .nlpca_net import NLPCANet
from .nlpca_utils import _compute_r2_vectorized, _conjugate_gradient_optimizer


def nlpca(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: Optional[int] = None,
    units_per_layer: Optional[list] = None,
    weight_decay: float = 0.001,
    weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> PCAResult:
    """
    Non-Linear Principal Component Analysis using neural networks.

    Args:
        data: Input tensor of shape (n_samples, n_features)
        n_components: Number of components to extract
        max_iter: Maximum iterations for optimization
        units_per_layer: Architecture of the neural network
        weight_decay: L2 regularization parameter
        weights: Initial weights (optional)
        verbose: Whether to print progress
        center: Whether to center the data
        scale: Whether to scale the data

    Returns:
        PCAResult object
    """
    train_out = data.T.contiguous()
    n_features, n_patterns = train_out.shape
    eps = 1e-8

    stds = torch.std(train_out, dim=1, unbiased=True)
    stds = torch.where(torch.isnan(stds) | (stds == 0), eps, stds)
    scaling_factor = (0.1 / torch.max(stds)).item()
    train_out = train_out * scaling_factor

    data_dist = (~torch.isnan(train_out)).float()
    train_out = torch.nan_to_num(train_out, 0.0)

    if max_iter is None:
        max_iter = 2 * n_features * n_patterns

    if units_per_layer is None:
        lh = (
            max(n_components, 10)
            if n_components >= 10
            else 2 + 2 * n_components
        )
        units_per_layer = [n_components, lh, n_features]

    w_num = sum(
        (1 + units_per_layer[i - 1]) * units_per_layer[i]
        for i in range(1, len(units_per_layer))
    )

    if weights is None:
        network_weights = 0.2 * (torch.rand(w_num) - 0.1)
        score_weights = torch.randn(n_components * n_patterns) * 0.1
        weights = torch.cat([score_weights, network_weights])

    net = NLPCANet(
        units_per_layer,
        weight_decay,
        data_dist,
        scaling_factor,
    )
    net.weights = weights

    if verbose:
        print(f"Training network with {max_iter} iterations...")

    final_weights = _conjugate_gradient_optimizer(
        net, train_out, max_iter, verbose
    )

    if verbose:
        print("Training network finished.")

    projection = final_weights[: n_components * n_patterns].view(
        n_patterns, n_components
    )
    net.weights = final_weights[n_components * n_patterns :]

    R2cum = _compute_r2_vectorized(data, projection, net, scaling_factor)

    result = PCAResult(
        transformed_data=projection,
        components=torch.zeros(0, data.shape[1]),
        eigenvalues=R2cum,
        method="nlpca",
        net=net,
    )

    return result
