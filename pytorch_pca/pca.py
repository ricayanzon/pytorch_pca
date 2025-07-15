from typing import Literal

import torch

from .methods import bpca, nipals, nlpca, ppca, rnipals, rpca, svd, svd_impute
from .pca_result import PCAResult
from .utils import _check_data, normalize_data

type AllowedMethod = Literal[
    "svd",
    "nipals",
    "rnipals",
    "ppca",
    "bpca",
    "svd_impute",
    "rpca",
    "nlpca",
]


def pca(
    data: torch.Tensor,
    method: AllowedMethod = "svd",
    n_components: int = 2,
    center: bool = True,
    scale: bool = False,
    **kwargs,
) -> PCAResult:
    """
    Unified interface for all PCA methods.

    Args:
        data: Input tensor of shape (n_samples, n_features)
        method: PCA method to use
        n_components: Number of components to extract
        center: Whether to center the data
        scale: Whether to scale the data
        **kwargs: Additional arguments for specific methods

    Returns:
        PCAResult object
    """
    _check_data(data)

    if n_components <= 0:
        raise ValueError("n_components must be a positive integer")
    if "max_iter" in kwargs and kwargs["max_iter"] <= 0:
        raise ValueError("max_iter must be positive")
    if "tolerance" in kwargs and kwargs["tolerance"] <= 0:
        raise ValueError("tolerance must be positive")

    methods = {
        "svd": svd,
        "nipals": nipals,
        "rnipals": rnipals,
        "ppca": ppca,
        "bpca": bpca,
        "svd_impute": svd_impute,
        "rpca": rpca,
        "nlpca": nlpca,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available methods: {list(methods.keys())}"
        )

    n_components = min(n_components, min(data.shape))
    if center or scale:
        data = normalize_data(data, center=center, scale=scale)

    return methods[method](data, n_components=n_components, **kwargs)
