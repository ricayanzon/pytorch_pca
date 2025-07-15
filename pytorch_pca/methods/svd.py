from warnings import warn

import torch

from ..pca_result import PCAResult
from .svd_impute import svd_impute


def svd(
    data: torch.Tensor,
    n_components: int = 2,
) -> PCAResult:
    """
    Standard SVD-based PCA implementation.
    If data has missing values, falls back to SVD with imputation.

    Args:
        data: Input tensor of shape (n_samples, n_features)
        n_components: Number of components to extract
        center: Whether to center the data
        scale: Whether to scale the data

    Returns:
        PCAResult object
    """

    if torch.isnan(data).any():
        warn(
            "Data contains missing values. Using SVD with imputation instead."
        )
        return svd_impute(data, n_components)

    U, S, Vt = torch.linalg.svd(data, full_matrices=False)
    components = Vt[:n_components]
    transformed_data = U[:, :n_components] * S[:n_components]
    explained_variance = S[:n_components] ** 2 / (data.shape[0] - 1)

    return PCAResult(transformed_data, components, explained_variance, "svd")
