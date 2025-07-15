import torch

from ..pca_result import PCAResult


def ppca(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 1000,
    tolerance: float = 1e-5,
) -> PCAResult:
    """
    Probabilistic PCA implementation using EM algorithm.
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
    missing_mask = torch.isnan(data)
    data[missing_mask] = 0

    W = (
        torch.randn(
            n_features, n_components, dtype=data.dtype, device=data.device
        )
        * 0.1
    )
    sigma2 = 1.0

    if not missing_mask.any():
        cov = data.T @ data / n_samples
        eigvals, eigvecs = torch.linalg.eigh(cov)
        idx = torch.argsort(eigvals, descending=True)
        W = eigvecs[:, idx[:n_components]] * eigvals[idx[:n_components]].sqrt()
        sigma2 = (
            max(eigvals[idx[n_components:]].mean().item(), 1e-6)
            if n_components < n_features
            else 1e-6
        )

    identity_matrix = torch.eye(
        n_components, dtype=data.dtype, device=data.device
    )

    for _ in range(max_iter):
        W_old = W.clone()

        SW = torch.zeros_like(W)
        SS = torch.zeros(
            n_components, n_components, dtype=data.dtype, device=data.device
        )
        sigma2_new = 0.0
        count = 0

        for i in range(n_samples):
            mask = ~missing_mask[i]
            n_obs = mask.sum()
            if n_obs == 0:
                continue

            W_i = W[mask]
            x_i = data[i, mask]

            M_i = W_i.T @ W_i + sigma2 * identity_matrix
            L_i = torch.linalg.cholesky(M_i)
            z_i = torch.cholesky_solve(W_i.T @ x_i.unsqueeze(1), L_i).squeeze()

            SW[mask] += x_i.unsqueeze(1) * z_i
            SS += torch.outer(z_i, z_i) + sigma2 * torch.cholesky_inverse(L_i)

            residual = x_i - W_i @ z_i
            sigma2_new += residual.dot(residual).item()
            count += n_obs.item()

        W = torch.linalg.solve(SS.T, SW.T).T
        sigma2 = max(sigma2_new / count, 1e-6) if count > 0 else sigma2

        if torch.norm(W - W_old) < tolerance:
            break

    M = W.T @ W + sigma2 * identity_matrix
    transformed_data = data @ torch.linalg.solve(M.T, W.T).T
    explained_variance = torch.diag(W.T @ W)

    return PCAResult(transformed_data, W.T, explained_variance, "ppca")
