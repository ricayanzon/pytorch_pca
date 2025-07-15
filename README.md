# PyTorch PCA

A comprehensive PCA implementation using PyTorch, inspired by the R package [pcaMethods](https://github.com/bioc/pcaMethods).

## Overview

This package provides a unified interface to eight PCA algorithms, all accessible via the `pca` function. The main entry point is `pytorch_pca.pca`, and the package exposes `pca`, `PCAResult`, and `AllowedMethod` in its public API.

[![Release](https://img.shields.io/github/v/tag/ricayanzon/pytorch_pca?label=Pypi&logo=pypi&logoColor=yellow)](https://pypi.org/project/pytorch_pca/)
![PythonVersion](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-informational)
![PytorchVersion](https://img.shields.io/badge/pytorch-2.7.1-blue)

## Installation

```bash
pip install pytorch_pca
```

## Features

This package provides 8 different PCA algorithms optimized for various scenarios:

- **`svd`**: Standard SVD-based PCA (fastest, complete data only)
- **`nipals`**: NIPALS algorithm (handles missing values effectively)
- **`rnipals`**: Robust NIPALS (resistant to outliers and missing values)
- **`ppca`**: Probabilistic PCA (classical probabilistic model)
- **`bpca`**: Bayesian PCA (probabilistic approach with uncertainty quantification)
- **`svd_impute`**: SVD-based PCA with missing value imputation
- **`rpca`**: Robust PCA using iterative outlier detection
- **`nlpca`**: Non-linear PCA using autoencoder neural network architecture

## Quick Start

```python
import torch
from pytorch_pca import pca

# Generate sample data
X = torch.randn(100, 20)

# Basic PCA using the unified interface
result = pca(X, method="svd", n_components=5)
print(f"Transformed data shape: {result.transformed_data.shape}")
print(f"Components shape: {result.components.shape}")
print(f"Explained variance: {result.explained_variance_ratio}")

# Alternative: use pcaMethods-style naming
print(f"Scores shape: {result.scores.shape}")
print(f"Loadings shape: {result.loadings.shape}")

# Handle missing data with NIPALS
X_missing = X.clone()
X_missing[10:20, 5:10] = float('nan')
result = pca(X_missing, method="nipals", n_components=3)

# Robust PCA for data with outliers
result = pca(X, method="rpca", n_components=3)

# Probabilistic approaches
result = pca(X, method="ppca", n_components=3)
result = pca(X, method="bpca", n_components=3)

# Non-linear PCA with neural networks
result = pca(X, method="nlpca", n_components=3)

# Reconstruct data
X_reconstructed = result.reconstruct(n_components=3)
```

## API Reference

### Unified Interface

```python
result = pca(data, method="svd", n_components=2, center=True, scale=False, **kwargs)
```

### Method-Specific Parameters

#### NIPALS Methods
```python
nipals(data, max_iter=1000, tol=1e-6, ...)
rnipals(data, max_iter=1000, tol=1e-6, ...)
```

#### Probabilistic Methods
```python
ppca(data, max_iter=1000, tol=1e-6, ...)
bpca(data, max_iter=1000, tol=1e-6, ...)
```

#### Robust PCA
```python
rpca(data, max_iter=100, tol=1e-6, ...)
```

#### Non-linear PCA
```python
nlpca(data, hidden_dims=[10, 5], max_iter=1000, lr=0.01, ...)
```

### Result Object

The `PCAResult` object provides:

- `transformed_data`: Data projected onto principal components `(n_samples, n_components)`
- `components`: Principal components (eigenvectors) `(n_components, n_features)`
- `eigenvalues`: Eigenvalues of the covariance matrix
- `explained_variance_ratio`: Proportion of variance explained by each component
- `method`: Name of the method used
- `scores`: Alias for `transformed_data` (pcaMethods compatibility)
- `loadings`: Transposed components (pcaMethods compatibility)
- `reconstruct(n_components=None)`: Reconstruct data using selected components

## Method Selection Guide

- **Complete data, speed priority**: `svd`
- **Missing values**: `nipals` or `svd_impute`
- **Outliers present**: `rpca` or `rnipals`
- **Uncertainty quantification**: `bpca`
- **Probabilistic modeling**: `ppca`
- **Non-linear relationships**: `nlpca`

## Dependencies

- **torch**: The only required dependency (>= 2.7.1)

## Development & Testing

- **black**: Code formatting
- **flake8**: Code linting
- **mypy**: Type checking
- **pytest**: Test runner
- **scikit-learn**: For test comparisons
- **setuptools**, **setuptools-scm**: Packaging

Comprehensive tests are provided in the `tests/` directory, covering all algorithms, edge cases, and robust/non-linear PCA scenarios.

## Testing

```bash
# Run all tests from root
pytest ./tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Ricardo Yanzon** - [ricayanzon](https://github.com/ricayanzon)

## Acknowledgments

- Inspired by the R package [pcaMethods](https://github.com/bioc/pcaMethods)
- Built with [PyTorch](https://pytorch.org/)