from typing import Optional

import torch

from .nl_pca_net import NLPCANet


class PCAResult:
    """Container for PCA results using standard scientific terminology.

    Attributes:
        components: Principal components (eigenvectors),
            shape (n_components, n_features)
        transformed_data: Data projected onto the principal components,
            shape (n_samples, n_components)
        eigenvalues: Eigenvalues of the covariance matrix
        explained_variance_ratio: Proportion of total variance explained by
            each component
        method: Name of the PCA method used
    """

    def __init__(
        self,
        transformed_data: torch.Tensor,
        components: torch.Tensor,
        eigenvalues: torch.Tensor,
        method: str,
        net: Optional[NLPCANet] = None,
    ):
        self.transformed_data = transformed_data
        self.components = components
        self.eigenvalues = eigenvalues
        self.explained_variance_ratio = (
            eigenvalues / eigenvalues.sum()
            if eigenvalues.numel() > 0
            else torch.empty(0)
        )
        self.method = method
        self.net = net

    @property
    def scores(self):
        """Transformed data (pcaMethods naming)."""
        return self.transformed_data

    @property
    def loadings(self):
        """Components transposed (pcaMethods naming)."""
        return self.components.T

    def reconstruct(
        self,
        n_components: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Reconstruct data using the fitted PCA model.

        Args:
            n_components: Number of components to use for reconstruction.
                If None, uses all available components.

        Returns:
            Reconstructed data tensor
        """
        if n_components is None:
            n_components = self.transformed_data.shape[1]

        if n_components <= 0 or n_components > max(
            self.components.shape[0], self.components.shape[1]
        ):
            raise ValueError(
                "n_components must be between 1 and "
                f"{max(self.components.shape[0], self.components.shape[1])}"
            )

        if self.method == "nlpca":
            if self.net is None:
                raise ValueError(
                    "Neural network data not available for "
                    "NLPCA reconstruction"
                )
            return self._reconstruct_nlpca(n_components)
        else:
            return self._reconstruct_linear(n_components)

    def _reconstruct_linear(self, n_components: int) -> torch.Tensor:
        """Reconstruct data using linear PCA methods."""
        transformed_data_subset = self.transformed_data[:, :n_components]
        components_subset = self.components[:n_components]
        return transformed_data_subset @ components_subset

    def _reconstruct_nlpca(self, n_components: int) -> torch.Tensor:
        """Reconstruct data using non-linear PCA neural network."""

        if self.net is None or self.net.weights is None:
            raise ValueError("Network weights are not initialized")

        transformed_data_subset = self.transformed_data[:, :n_components]
        weights_list = []
        biases_list = []

        for start, w_end, b_end in self.net.layer_indices:
            n_in = self.net.units_per_layer[len(weights_list)]
            n_out = self.net.units_per_layer[len(weights_list) + 1]

            w = self.net.weights[start:w_end].view(n_out, n_in)
            b = self.net.weights[w_end:b_end]
            weights_list.append(w)
            biases_list.append(b)

        h = transformed_data_subset
        w_first = weights_list[0][:, :n_components]
        h = torch.nn.functional.linear(h, w_first, biases_list[0])
        h = torch.tanh(h)

        for j in range(1, len(weights_list)):
            h = torch.nn.functional.linear(h, weights_list[j], biases_list[j])
            h = torch.tanh(h)

        return h / self.net.scaling_factor
