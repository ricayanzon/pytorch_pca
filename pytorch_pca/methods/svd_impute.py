import torch

from ..pca_result import PCAResult


def svd_impute(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 100,
    tolerance: float = 0.01,
) -> PCAResult:
    """
    SVD-based missing value imputation followed by PCA.

    Args:
        data: Input tensor of shape (n_samples, n_features)
        n_components: Number of components to extract
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance
        center: Whether to center the data
        scale: Whether to scale the data

    Returns:
        PCAResult object
    """

    n_samples, n_features = data.shape
    missing_mask = torch.isnan(data)
    has_missing = missing_mask.any()

    if has_missing:
        col_means = torch.nanmean(data, dim=0)
        col_means[torch.isnan(col_means)] = 0.0
        X_filled = data.clone()
        for j in range(n_features):
            X_filled[missing_mask[:, j], j] = col_means[j]

        prev_mse = float("inf")
        for _ in range(max_iter):
            U, S, Vt = torch.linalg.svd(X_filled, full_matrices=False)

            k = min(n_components, len(S))
            X_reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :]

            X_filled = torch.where(missing_mask, X_reconstructed, data)

            mse = torch.mean(
                (X_filled[missing_mask] - X_reconstructed[missing_mask]) ** 2
            )
            if torch.abs(mse - prev_mse) < tolerance:
                break
            prev_mse = mse
    else:
        X_filled = data

    U, S, Vt = torch.linalg.svd(X_filled, full_matrices=False)

    k = min(n_components, len(S))
    components = Vt[:k]
    transformed_data = U[:, :k] * S[:k]
    eigenvalues = (S[:k] ** 2) / (n_samples - 1)

    return PCAResult(transformed_data, components, eigenvalues, "svd_impute")
