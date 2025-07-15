import torch

from ..pca_result import PCAResult


def rpca(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
) -> PCAResult:
    """
    Robust PCA implementation using L1 norm.

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
    U = torch.zeros(
        n_samples, n_components, dtype=data.dtype, device=data.device
    )
    V = torch.zeros(
        n_features, n_components, dtype=data.dtype, device=data.device
    )
    S = torch.zeros(n_components, dtype=data.dtype, device=data.device)

    data_work = data.clone()
    eps = 1e-10

    for k in range(n_components):
        u = torch.median(torch.abs(data_work), dim=1).values
        u = u / (torch.norm(u) + eps)
        v = torch.zeros(n_features, dtype=data.dtype, device=data.device)

        for _ in range(max_iter):
            u_old = u.clone()

            weights_u = torch.abs(u).clamp(min=eps)
            v = (data_work.T @ (u * weights_u)) / torch.sum(weights_u**2)
            v = v / (torch.norm(v) + eps)

            weights_v = torch.abs(v).clamp(min=eps)
            u = (data_work @ (v * weights_v)) / torch.sum(weights_v**2)
            u = u / (torch.norm(u) + eps)

            if torch.norm(u - u_old) < tolerance:
                break

        s = torch.dot(u, data_work @ v)

        U[:, k] = u
        V[:, k] = v
        S[k] = s

        data_work -= s * torch.outer(u, v)

    transformed_data = U * S
    explained_variance = S**2 / (n_samples - 1)

    return PCAResult(transformed_data, V.T, explained_variance, "rpca")
