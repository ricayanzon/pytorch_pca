import torch

from ..pca_result import PCAResult


def rnipals(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
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
    n_samples, n_features = data.shape
    weights = _compute_hubert_weights(data)
    transformed_data = torch.zeros(n_samples, n_components, device=data.device)
    components = torch.zeros(n_components, n_features, device=data.device)
    explained_variance = torch.zeros(n_components, device=data.device)

    data_residual = data.clone()

    for n in range(n_components):
        mask = ~torch.isnan(data_residual)
        weighted_squared = torch.where(
            mask,
            weights.unsqueeze(1) * data_residual**2,
            torch.zeros_like(data_residual),
        )
        col_vars = weighted_squared.sum(dim=0)

        if col_vars.max() < 1e-10:
            break

        index_of_max_variance = torch.argmax(col_vars)
        projected_data = data_residual[:, index_of_max_variance].clone()
        component_vector = torch.zeros(n_features, device=data.device)

        for _ in range(max_iter):
            projected_data_old = projected_data.clone()

            valid_mask = ~torch.isnan(data_residual) & ~torch.isnan(
                projected_data
            ).unsqueeze(1)

            weighted_X = torch.where(
                valid_mask,
                weights.unsqueeze(1)
                * data_residual
                * projected_data.unsqueeze(1),
                torch.zeros_like(data_residual),
            )
            weighted_scores_sq = torch.where(
                valid_mask,
                weights.unsqueeze(1) * (projected_data.unsqueeze(1) ** 2),
                torch.zeros_like(data_residual),
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

            valid_features = ~torch.isnan(data_residual)
            X_masked = torch.where(
                valid_features, data_residual, torch.zeros_like(data_residual)
            )
            comp_masked = torch.where(
                valid_features,
                component_vector.unsqueeze(0),
                torch.zeros_like(data_residual),
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

        valid_mask = ~torch.isnan(data_residual) & ~torch.isnan(
            projected_data
        ).unsqueeze(1)
        data_residual = torch.where(
            valid_mask,
            data_residual
            - projected_data.unsqueeze(1) * component_vector.unsqueeze(0),
            data_residual,
        )

        valid_scores = ~torch.isnan(projected_data)
        if valid_scores.sum() > 1:
            explained_variance[n] = torch.sum(
                weights[valid_scores] * (projected_data[valid_scores] ** 2)
            ) / torch.sum(weights[valid_scores])
        else:
            explained_variance[n] = 0.0

        if n < n_components - 1:
            weights = _compute_hubert_weights(data_residual)

    return PCAResult(
        transformed_data, components, explained_variance, "rnipals"
    )


def _compute_hubert_weights(X: torch.Tensor, c: float = 2.5) -> torch.Tensor:
    """Compute Hubert weights for robust estimation."""

    n_features = X.shape[1]
    valid_mask = ~torch.isnan(X)

    if hasattr(torch, "nanmedian"):
        center = torch.nanmedian(X, dim=0).values
    else:
        center = torch.zeros(n_features, device=X.device, dtype=X.dtype)
        for j in range(n_features):
            if valid_mask[:, j].any():
                center[j] = torch.median(X[valid_mask[:, j], j])

    deviations = X - center.unsqueeze(0)
    abs_deviations = torch.abs(deviations)
    mad = torch.zeros(n_features, device=X.device, dtype=X.dtype)
    for j in range(n_features):
        if valid_mask[:, j].sum() > 1:
            valid_abs_devs = abs_deviations[valid_mask[:, j], j]
            mad[j] = 1.4826 * torch.median(valid_abs_devs)
        else:
            mad[j] = 1.0

    mad = torch.clamp(mad, min=1e-6)

    standardized_devs = torch.where(
        valid_mask, deviations / mad.unsqueeze(0), torch.zeros_like(deviations)
    )

    n_valid = valid_mask.sum(dim=1).float()
    n_valid = torch.clamp(n_valid, min=1.0)
    distances = torch.sqrt((standardized_devs**2).sum(dim=1) / n_valid)

    weights = torch.where(
        distances <= c, torch.ones_like(distances), c / distances
    )

    all_nan = ~valid_mask.any(dim=1)
    weights = torch.where(all_nan, torch.zeros_like(weights), weights)

    return weights
