import torch

from ..pca_result import PCAResult


def bpca(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 100,
    tolerance: float = 1e-4,
) -> PCAResult:
    """
    Bayesian PCA implementation.
    Can handle missing values.

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
    missing = torch.isnan(data)
    data = torch.nan_to_num(data, 0.0)

    W = torch.randn(n_features, n_components, dtype=data.dtype) * 0.01
    alpha = torch.ones(n_components, dtype=data.dtype)
    sigma2 = torch.tensor(1.0, dtype=data.dtype)

    for _ in range(max_iter):
        sigma2_old = sigma2.clone()

        SW = torch.zeros_like(W)
        SS = torch.zeros(n_components, n_components, dtype=data.dtype)

        for i in range(n_samples):
            mask = ~missing[i]
            if not mask.any():
                continue

            x = data[i, mask]
            w = W[mask]

            M = torch.linalg.inv(torch.diag(alpha) + w.T @ w / sigma2)
            z = M @ w.T @ x / sigma2

            SW[mask] += x.unsqueeze(1) * z
            SS += torch.outer(z, z) + M

        W = torch.linalg.solve(SS + sigma2 * torch.diag(alpha), SW.T).T

        alpha = n_features / (W**2).sum(dim=0)

        sse = torch.tensor(0.0, dtype=data.dtype)
        n_obs = torch.tensor(0, dtype=torch.int64)

        for i in range(n_samples):
            mask = ~missing[i]
            if not mask.any():
                continue

            x = data[i, mask]
            w = W[mask]

            M = torch.linalg.inv(torch.diag(alpha) + w.T @ w / sigma2)
            z = M @ w.T @ x / sigma2

            sse += ((x - w @ z) ** 2).sum() + sigma2 * torch.trace(w @ M @ w.T)
            n_obs += mask.sum()

        sigma2 = sse / n_obs.float()

        if torch.abs(torch.log(sigma2) - torch.log(sigma2_old)) < tolerance:
            break

    Z = torch.zeros(n_samples, n_components, dtype=data.dtype)
    for i in range(n_samples):
        mask = ~missing[i]
        if mask.any():
            w = W[mask]
            M = torch.linalg.inv(torch.diag(alpha) + w.T @ w / sigma2)
            Z[i] = M @ w.T @ data[i, mask] / sigma2

    eigenvalues = (W.T @ W).diag() / n_samples

    return PCAResult(Z, W.T, eigenvalues, "bpca")
