from typing import Optional
from warnings import warn

import torch

from .nl_pca_net import NLPCANet
from .pca_result import PCAResult
from .utils import _compute_hubert_weights, normalize_data


def svd(
    data: torch.Tensor,
    n_components: int = 2,
    center: bool = True,
    scale: bool = False,
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
        return svd_impute(data, n_components, center=center, scale=scale)

    n_components = min(n_components, min(data.shape))
    X = normalize_data(data, center=center, scale=scale)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    components = Vt[:n_components]
    transformed_data = U[:, :n_components] * S[:n_components]
    explained_variance = S[:n_components] ** 2 / (data.shape[0] - 1)

    return PCAResult(transformed_data, components, explained_variance, "svd")


def nipals(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
    center: bool = True,
    scale: bool = False,
) -> PCAResult:
    """
    NIPALS (Non-linear Iterative Partial Least Squares) PCA implementation.
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

    n_components = min(n_components, min(data.shape))
    X = normalize_data(data, center=center, scale=scale)
    n_samples, n_features = X.shape
    transformed_data = torch.zeros(n_samples, n_components)
    components = torch.zeros(n_components, n_features)
    explained_variance = torch.zeros(n_components)

    X_residual = X.clone()

    for n in range(n_components):
        col_vars = torch.zeros(n_features)
        for j in range(n_features):
            mask = ~torch.isnan(X_residual[:, j])
            if mask.sum() > 0:
                col_vars[j] = torch.sum(X_residual[mask, j] ** 2)

        if col_vars.max() < 1e-10:
            break

        index_of_max_variance = torch.argmax(col_vars)
        projected_data = X_residual[:, index_of_max_variance].clone()
        component_vector = torch.zeros(n_features)

        for _ in range(max_iter):
            valid_scores = ~torch.isnan(projected_data)

            for j in range(n_features):
                mask = ~torch.isnan(X_residual[:, j]) & valid_scores
                if mask.sum() > 0:
                    numerator = torch.sum(
                        X_residual[mask, j] * projected_data[mask]
                    )
                    denominator = torch.sum(projected_data[mask] ** 2) + 1e-10
                    component_vector[j] = numerator / denominator
                else:
                    component_vector[j] = 0.0

            component_vector_norm = torch.norm(component_vector)
            if component_vector_norm > 1e-10:
                component_vector = component_vector / component_vector_norm
            else:
                break

            projected_data_old = projected_data.clone()

            for i in range(n_samples):
                mask = ~torch.isnan(X_residual[i, :])
                if mask.sum() > 0:
                    projected_data[i] = torch.sum(
                        X_residual[i, mask] * component_vector[mask]
                    )
                else:
                    projected_data[i] = 0.0

            valid_scores = (projected_data != 0.0) | (
                projected_data_old != 0.0
            )
            if valid_scores.sum() > 0:
                diff = torch.norm(
                    projected_data[valid_scores]
                    - projected_data_old[valid_scores]
                )
                if diff < tolerance:
                    break

        transformed_data[:, n] = projected_data
        components[n, :] = component_vector

        for i in range(n_samples):
            for j in range(n_features):
                if (
                    not torch.isnan(X_residual[i, j])
                    and projected_data[i] != 0.0
                ):
                    X_residual[i, j] -= projected_data[i] * component_vector[j]

        valid_scores = projected_data != 0.0
        if valid_scores.sum() > 1:
            explained_variance[n] = torch.sum(
                projected_data[valid_scores] ** 2
            ) / (valid_scores.sum() - 1)
        else:
            explained_variance[n] = 0.0

    return PCAResult(
        transformed_data, components, explained_variance, "nipals"
    )


def rnipals(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
    center: bool = True,
    scale: bool = False,
) -> PCAResult:
    """
    Robust NIPALS PCA implementation.

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

    n_components = min(n_components, min(data.shape))
    X = normalize_data(data, center=center, scale=scale)
    n_samples, n_features = X.shape

    weights = _compute_hubert_weights(X)

    transformed_data = torch.zeros(n_samples, n_components, device=X.device)
    components = torch.zeros(n_components, n_features, device=X.device)
    explained_variance = torch.zeros(n_components, device=X.device)

    X_residual = X.clone()

    for n in range(n_components):
        mask = ~torch.isnan(X_residual)
        weighted_squared = torch.where(
            mask,
            weights.unsqueeze(1) * X_residual**2,
            torch.zeros_like(X_residual),
        )
        col_vars = weighted_squared.sum(dim=0)

        if col_vars.max() < 1e-10:
            break

        index_of_max_variance = torch.argmax(col_vars)
        projected_data = X_residual[:, index_of_max_variance].clone()
        component_vector = torch.zeros(n_features, device=X.device)

        for _ in range(max_iter):
            projected_data_old = projected_data.clone()

            valid_mask = ~torch.isnan(X_residual) & ~torch.isnan(
                projected_data
            ).unsqueeze(1)

            weighted_X = torch.where(
                valid_mask,
                weights.unsqueeze(1)
                * X_residual
                * projected_data.unsqueeze(1),
                torch.zeros_like(X_residual),
            )
            weighted_scores_sq = torch.where(
                valid_mask,
                weights.unsqueeze(1) * (projected_data.unsqueeze(1) ** 2),
                torch.zeros_like(X_residual),
            )

            numerator = weighted_X.sum(dim=0)
            denominator = weighted_scores_sq.sum(dim=0) + 1e-10

            component_vector = torch.where(
                denominator > 1e-10,
                numerator / denominator,
                torch.zeros_like(numerator),
            )

            component_norm = torch.norm(component_vector)
            if component_norm > 1e-10:
                component_vector = component_vector / component_norm
            else:
                break

            valid_features = ~torch.isnan(X_residual)
            X_masked = torch.where(
                valid_features, X_residual, torch.zeros_like(X_residual)
            )
            comp_masked = torch.where(
                valid_features,
                component_vector.unsqueeze(0),
                torch.zeros_like(X_residual),
            )

            projected_data = (X_masked * comp_masked).sum(dim=1)
            all_nan_rows = (~valid_features).all(dim=1)
            projected_data = torch.where(
                all_nan_rows, torch.nan, projected_data
            )

            valid_scores = ~torch.isnan(projected_data)
            if valid_scores.sum() > 0:
                diff = torch.norm(
                    projected_data[valid_scores]
                    - projected_data_old[valid_scores]
                )
                if diff < tolerance:
                    break

        transformed_data[:, n] = projected_data
        components[n, :] = component_vector

        valid_mask = ~torch.isnan(X_residual) & ~torch.isnan(
            projected_data
        ).unsqueeze(1)
        X_residual = torch.where(
            valid_mask,
            X_residual
            - projected_data.unsqueeze(1) * component_vector.unsqueeze(0),
            X_residual,
        )

        valid_scores = ~torch.isnan(projected_data)
        if valid_scores.sum() > 1:
            explained_variance[n] = torch.sum(
                weights[valid_scores] * (projected_data[valid_scores] ** 2)
            ) / torch.sum(weights[valid_scores])
        else:
            explained_variance[n] = 0.0

        if n < n_components - 1:
            weights = _compute_hubert_weights(X_residual)

    return PCAResult(
        transformed_data, components, explained_variance, "rnipals"
    )


def ppca(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 1000,
    tolerance: float = 1e-5,
    center: bool = True,
    scale: bool = False,
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

    n_components = min(n_components, min(data.shape))

    X = normalize_data(data, center=center, scale=scale)
    n_samples, n_features = X.shape

    missing_mask = torch.isnan(X)
    X[missing_mask] = 0

    W = (
        torch.randn(n_features, n_components, dtype=X.dtype, device=X.device)
        * 0.1
    )
    sigma2 = 1.0

    if not missing_mask.any():
        cov = X.T @ X / n_samples
        eigvals, eigvecs = torch.linalg.eigh(cov)
        idx = torch.argsort(eigvals, descending=True)
        W = eigvecs[:, idx[:n_components]] * eigvals[idx[:n_components]].sqrt()
        sigma2 = (
            max(eigvals[idx[n_components:]].mean().item(), 1e-6)
            if n_components < n_features
            else 1e-6
        )

    identity_matrix = torch.eye(n_components, dtype=X.dtype, device=X.device)

    for _ in range(max_iter):
        W_old = W.clone()

        SW = torch.zeros_like(W)
        SS = torch.zeros(
            n_components, n_components, dtype=X.dtype, device=X.device
        )
        sigma2_new = 0.0
        count = 0

        for i in range(n_samples):
            mask = ~missing_mask[i]
            n_obs = mask.sum()
            if n_obs == 0:
                continue

            W_i = W[mask]
            x_i = X[i, mask]

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
    transformed_data = X @ torch.linalg.solve(M.T, W.T).T
    explained_variance = torch.diag(W.T @ W)

    return PCAResult(transformed_data, W.T, explained_variance, "ppca")


def bpca(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 100,
    tolerance: float = 1e-4,
    center: bool = True,
    scale: bool = False,
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

    n_components = min(n_components, min(data.shape))
    X = normalize_data(data, center=center, scale=scale)
    n_samples, n_features = X.shape

    missing = torch.isnan(X)
    X = torch.nan_to_num(X, 0.0)

    W = torch.randn(n_features, n_components, dtype=X.dtype) * 0.01
    alpha = torch.ones(n_components, dtype=X.dtype)
    sigma2 = torch.tensor(1.0, dtype=X.dtype)

    for _ in range(max_iter):
        sigma2_old = sigma2.clone()

        SW = torch.zeros_like(W)
        SS = torch.zeros(n_components, n_components, dtype=X.dtype)

        for i in range(n_samples):
            mask = ~missing[i]
            if not mask.any():
                continue

            x = X[i, mask]
            w = W[mask]

            M = torch.linalg.inv(torch.diag(alpha) + w.T @ w / sigma2)
            z = M @ w.T @ x / sigma2

            SW[mask] += x.unsqueeze(1) * z
            SS += torch.outer(z, z) + M

        W = torch.linalg.solve(SS + sigma2 * torch.diag(alpha), SW.T).T

        alpha = n_features / (W**2).sum(dim=0)

        sse = torch.tensor(0.0, dtype=X.dtype)
        n_obs = torch.tensor(0, dtype=torch.int64)

        for i in range(n_samples):
            mask = ~missing[i]
            if not mask.any():
                continue

            x = X[i, mask]
            w = W[mask]

            M = torch.linalg.inv(torch.diag(alpha) + w.T @ w / sigma2)
            z = M @ w.T @ x / sigma2

            sse += ((x - w @ z) ** 2).sum() + sigma2 * torch.trace(w @ M @ w.T)
            n_obs += mask.sum()

        sigma2 = sse / n_obs.float()

        if torch.abs(torch.log(sigma2) - torch.log(sigma2_old)) < tolerance:
            break

    Z = torch.zeros(n_samples, n_components, dtype=X.dtype)
    for i in range(n_samples):
        mask = ~missing[i]
        if mask.any():
            w = W[mask]
            M = torch.linalg.inv(torch.diag(alpha) + w.T @ w / sigma2)
            Z[i] = M @ w.T @ X[i, mask] / sigma2

    eigenvalues = (W.T @ W).diag() / n_samples

    return PCAResult(Z, W.T, eigenvalues, "bpca")


def svd_impute(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 100,
    tolerance: float = 0.01,
    center: bool = True,
    scale: bool = False,
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
    n_components = min(n_components, min(n_samples, n_features))

    X = normalize_data(data, center=center, scale=scale)

    missing_mask = torch.isnan(X)
    has_missing = missing_mask.any()

    if has_missing:
        col_means = torch.nanmean(X, dim=0)
        col_means[torch.isnan(col_means)] = 0.0
        X_filled = X.clone()
        for j in range(n_features):
            X_filled[missing_mask[:, j], j] = col_means[j]

        prev_mse = float("inf")
        for _ in range(max_iter):
            U, S, Vt = torch.linalg.svd(X_filled, full_matrices=False)

            k = min(n_components, len(S))
            X_reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :]

            X_filled = torch.where(missing_mask, X_reconstructed, X)

            mse = torch.mean(
                (X_filled[missing_mask] - X_reconstructed[missing_mask]) ** 2
            )
            if torch.abs(mse - prev_mse) < tolerance:
                break
            prev_mse = mse
    else:
        X_filled = X

    U, S, Vt = torch.linalg.svd(X_filled, full_matrices=False)

    k = min(n_components, len(S))
    components = Vt[:k]
    transformed_data = U[:, :k] * S[:k]
    eigenvalues = (S[:k] ** 2) / (n_samples - 1)

    return PCAResult(transformed_data, components, eigenvalues, "svd_impute")


def rpca(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
    center: bool = True,
    scale: bool = False,
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

    n_components = min(n_components, min(data.shape))

    X = normalize_data(data, center=center, scale=scale)
    n_samples, n_features = X.shape

    U = torch.zeros(n_samples, n_components, dtype=X.dtype, device=X.device)
    V = torch.zeros(n_features, n_components, dtype=X.dtype, device=X.device)
    S = torch.zeros(n_components, dtype=X.dtype, device=X.device)

    X_work = X.clone()
    eps = 1e-10

    for k in range(n_components):
        u = torch.median(torch.abs(X_work), dim=1).values
        u = u / (torch.norm(u) + eps)
        v = torch.zeros(n_features, dtype=X.dtype, device=X.device)

        for _ in range(max_iter):
            u_old = u.clone()

            weights_u = torch.abs(u).clamp(min=eps)
            v = (X_work.T @ (u * weights_u)) / torch.sum(weights_u**2)
            v = v / (torch.norm(v) + eps)

            weights_v = torch.abs(v).clamp(min=eps)
            u = (X_work @ (v * weights_v)) / torch.sum(weights_v**2)
            u = u / (torch.norm(u) + eps)

            if torch.norm(u - u_old) < tolerance:
                break

        s = torch.dot(u, X_work @ v)

        U[:, k] = u
        V[:, k] = v
        S[k] = s

        X_work -= s * torch.outer(u, v)

    transformed_data = U * S
    explained_variance = S**2 / (n_samples - 1)

    return PCAResult(transformed_data, V.T, explained_variance, "rpca")


def _conjugate_gradient_optimizer(net, train_out, max_iter, verbose):
    _, n_patterns = train_out.shape
    n_components = net.units_per_layer[0]
    eps = 1e-8

    weights = net.weights.clone().requires_grad_(True)

    prev_grad = None
    search_direction = None

    alpha_init = 0.01
    alpha_decay = 0.001

    for iteration in range(max_iter):
        scores = weights[: n_components * n_patterns].view(
            n_patterns, n_components
        )

        h = scores
        offset = n_components * n_patterns

        for i, (start, w_end, b_end) in enumerate(net.layer_indices):
            n_in = net.units_per_layer[i]
            n_out = net.units_per_layer[i + 1]

            w = weights[offset + start : offset + w_end].view(n_out, n_in)
            b = weights[offset + w_end : offset + b_end]

            h = torch.nn.functional.linear(h, w, b)
            h = torch.tanh(h)

        output = h.T

        error = (output - train_out) * net.data_dist
        mse = torch.sum(error**2) / (torch.sum(net.data_dist) + eps)

        weight_penalty = net.weight_decay * torch.sum(
            weights[n_components * n_patterns :] ** 2
        )
        total_loss = mse + weight_penalty

        grad = torch.autograd.grad(total_loss, weights, create_graph=False)[0]

        if iteration == 0:
            search_direction = -grad
        else:
            if prev_grad is not None and search_direction is not None:
                beta = torch.clamp(
                    torch.sum(grad**2) / (torch.sum(prev_grad**2) + eps),
                    0,
                    10,
                )
                search_direction = -grad + beta * search_direction
            else:
                search_direction = -grad

        alpha = alpha_init / (1 + iteration * alpha_decay)

        with torch.no_grad():
            weights = weights + alpha * search_direction
            weights.requires_grad_(True)

        prev_grad = grad.detach()

        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {total_loss.item():.6f}")

    return weights


def _compute_r2_vectorized(data, scores, net, scaling_factor):
    n_components = scores.shape[1]
    valid_data = torch.nan_to_num(data, 0.0)
    data_mask = ~torch.isnan(data)
    TSS = torch.sum(valid_data**2 * data_mask.float())
    R2cum = torch.zeros(n_components)

    weights_list = []
    biases_list = []

    if net.weights is None:
        raise ValueError("Network weights are not initialized")

    for start, w_end, b_end in net.layer_indices:
        n_in = net.units_per_layer[len(weights_list)]
        n_out = net.units_per_layer[len(weights_list) + 1]

        w = net.weights[start:w_end].view(n_out, n_in)
        b = net.weights[w_end:b_end]
        weights_list.append(w)
        biases_list.append(b)

    for i in range(1, n_components + 1):
        scores_subset = scores[:, :i]
        h = scores_subset

        w_first = weights_list[0][:, :i]
        h = torch.nn.functional.linear(h, w_first, biases_list[0])
        h = torch.tanh(h)

        for j in range(1, len(weights_list)):
            h = torch.nn.functional.linear(h, weights_list[j], biases_list[j])
            h = torch.tanh(h)

        reconstructed = h / scaling_factor
        diff = valid_data - reconstructed
        RSS = torch.sum((diff**2) * data_mask.float())

        if TSS > 1e-10:
            R2cum[i - 1] = torch.clamp(1 - RSS / TSS, 0.0, 1.0)
        else:
            R2cum[i - 1] = 0.0

    return R2cum


def nlpca(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: Optional[int] = None,
    units_per_layer: Optional[list] = None,
    weight_decay: float = 0.001,
    weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    center: bool = True,
    scale: bool = False,
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

    if center or scale:
        data = normalize_data(data, center=center, scale=scale)

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
