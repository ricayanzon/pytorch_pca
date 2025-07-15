import torch

from ..pca_result import PCAResult


def nipals(
    data: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
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
    n_samples, n_features = data.shape
    transformed_data = torch.zeros(n_samples, n_components)
    components = torch.zeros(n_components, n_features)
    explained_variance = torch.zeros(n_components)

    data_residual = data.clone()

    for n in range(n_components):
        col_vars = torch.zeros(n_features)
        for j in range(n_features):
            mask = ~torch.isnan(data_residual[:, j])
            if mask.sum() > 0:
                col_vars[j] = torch.sum(data_residual[mask, j] ** 2)

        if col_vars.max() < 1e-10:
            break

        index_of_max_variance = torch.argmax(col_vars)
        projected_data = data_residual[:, index_of_max_variance].clone()
        component_vector = torch.zeros(n_features)

        for _ in range(max_iter):
            valid_scores = ~torch.isnan(projected_data)

            for j in range(n_features):
                mask = ~torch.isnan(data_residual[:, j]) & valid_scores
                if mask.sum() > 0:
                    numerator = torch.sum(
                        data_residual[mask, j] * projected_data[mask]
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
                mask = ~torch.isnan(data_residual[i, :])
                if mask.sum() > 0:
                    projected_data[i] = torch.sum(
                        data_residual[i, mask] * component_vector[mask]
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
                    not torch.isnan(data_residual[i, j])
                    and projected_data[i] != 0.0
                ):
                    data_residual[i, j] -= (
                        projected_data[i] * component_vector[j]
                    )

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
