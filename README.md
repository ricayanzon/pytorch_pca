# PyTorch PCA

A comprehensive PCA implementation using PyTorch, inspired by the R package [pcaMethods](https://github.com/bioc/pcaMethods).

[![Release](https://img.shields.io/github/v/tag/ricayanzon/pytorch_pca?label=Pypi&logo=pypi&logoColor=yellow)](https://pypi.org/project/pytorch_pca/)
![PythonVersion](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-informational)
![PytorchVersion](https://img.shields.io/badge/pytorch-2.7.1-blue)

[![Black_logo](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Flake8](https://github.com/ricayanzon/pytorch_pca/actions/workflows/flake.yaml/badge.svg)](https://github.com/ricayanzon/pytorch_pca/actions/workflows/flake.yaml)
[![MyPy](https://github.com/ricayanzon/pytorch_pca/actions/workflows/mypy.yaml/badge.svg)](https://github.com/ricayanzon/pytorch_pca/actions/workflows/mypy.yaml)
[![PyLint](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ricayanzon/8fb4f3f78584e085dd7b0cca7e046d1f/raw/pytorch_pca_pylint.json)](https://github.com/ricayanzon/pytorch_pca/actions/workflows/pylint.yaml)

[![Tests](https://github.com/ricayanzon/pytorch_pca/actions/workflows/tests.yaml/badge.svg)](https://github.com/ricayanzon/pytorch_pca/actions/workflows/tests.yaml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ricayanzon/c5a6b5731db93da673f8e258b2669080/raw/pytorch_pca_tests.json)](https://github.com/ricayanzon/pytorch_pca/actions/workflows/tests.yaml)

## Installation

```bash
pip install torch_pca
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
from torch_pca import pca

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

## Dependencies

- **torch**: The only required dependency (>= 2.7.1)
- **scikit-learn**: Optional, only needed for comparison in tests

## Development Dependencies

- **black**: Code formatting
- **flake8**: Code linting
- **mypy**: Type checking
- **setuptools**: Package building
- **setuptools-scm**: Version management

## Testing

```bash
# Run all tests
python test_all.py
```

## API Reference

### Unified Interface

```python
result = pca(data, method="svd", n_components=2, center=True, scale=False, **kwargs)
```

### Individual Methods

All methods follow a consistent interface:

```python
result = method_name(data, n_components, center=True, scale=False, **kwargs)
```

**Parameters:**
- `data`: Input tensor of shape `(n_samples, n_features)`
- `n_components`: Number of principal components to extract
- `center`: Whether to center the data (default: `True`)
- `scale`: Whether to scale to unit variance (default: `False`)
- `**kwargs`: Method-specific parameters

**Returns:**
- `PCAResult` object with the following attributes:
  - `transformed_data`: Data projected onto principal components `(n_samples, n_components)`
  - `components`: Principal components (eigenvectors) `(n_components, n_features)`
  - `eigenvalues`: Eigenvalues of the covariance matrix
  - `explained_variance_ratio`: Proportion of variance explained by each component
  - `method`: Name of the method used
  - `scores`: Alias for `transformed_data` (pcaMethods compatibility)
  - `loadings`: Transposed components (pcaMethods compatibility)

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

## Method Selection Guide

- **Complete data, speed priority**: `svd`
- **Missing values**: `nipals` or `svd_impute`
- **Outliers present**: `rpca` or `rnipals`
- **Uncertainty quantification**: `bpca`
- **Probabilistic modeling**: `ppca`
- **Non-linear relationships**: `nlpca`
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Ricardo Yanzon** - [ricayanzon](https://github.com/ricayanzon)

## Acknowledgments

- Inspired by the R package [pcaMethods](https://github.com/bioc/pcaMethods)
- Built with [PyTorch](https://pytorch.org/)