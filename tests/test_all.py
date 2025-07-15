import pytest
import torch
from sklearn.decomposition import PCA

from pytorch_pca import AllowedMethod, PCAResult, pca


def generate_test_data(
    n_samples=100,
    n_features=10,
    noise_level=0.1,
    missing_ratio=0.0,
    add_outliers=False,
    seed=42,
):
    """Generate test data with known structure."""
    torch.manual_seed(seed)

    # Generate data with known principal components
    true_scores = torch.randn(n_samples, 3)
    true_loadings = torch.randn(n_features, 3)
    data = true_scores @ true_loadings.T

    # Add noise
    if noise_level > 0:
        data += torch.randn_like(data) * noise_level

    # Add missing values
    if missing_ratio > 0:
        missing_mask = torch.rand_like(data) < missing_ratio
        data[missing_mask] = float("nan")

    # Add outliers
    if add_outliers:
        outlier_mask = torch.rand(n_samples) < 0.05
        data[outlier_mask] *= 10

    return data


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def basic_data():
    """Generate basic test data."""
    return generate_test_data()


@pytest.fixture
def missing_data():
    """Generate test data with missing values."""
    return generate_test_data(missing_ratio=0.2)


@pytest.fixture
def outlier_data():
    """Generate test data with outliers."""
    return generate_test_data(add_outliers=True)


@pytest.fixture(
    params=[
        "svd",
        "nipals",
        "rnipals",
        "ppca",
        "bpca",
        "svd_impute",
        "rpca",
        "nlpca",
    ]
)
def method(request):
    """All PCA methods."""
    return request.param


@pytest.fixture(
    params=["nipals", "rnipals", "ppca", "bpca", "svd_impute", "nlpca"]
)
def missing_method(request):
    """Methods that handle missing values."""
    return request.param


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def validate_pca_result(result, data, method, n_components):
    """Validate PCA result structure and values."""
    n_samples, n_features = data.shape

    # Validate result structure
    assert isinstance(result, PCAResult)
    assert result.method == method
    assert result.scores.shape == (n_samples, n_components)

    # NLPCA doesn't have traditional loadings
    if method == "nlpca":
        assert result.loadings.shape == (n_features, 0)
    else:
        assert result.loadings.shape == (n_features, n_components)

    assert result.eigenvalues.shape == (n_components,)

    # Validate finiteness
    assert torch.isfinite(result.scores).all()
    assert torch.isfinite(result.eigenvalues).all()
    if method != "nlpca":
        assert torch.isfinite(result.loadings).all()


def create_special_data(pattern="row_missing"):
    """Create data with special patterns for edge case testing."""
    patterns = {
        "row_missing": lambda: torch.cat(
            [torch.full((1, 5), float("nan")), torch.randn(19, 5)]
        ),
        "column_missing": lambda: torch.cat(
            [torch.full((20, 1), float("nan")), torch.randn(20, 4)], dim=1
        ),
        "high_missing": lambda: _add_missing(torch.randn(50, 10), 0.7),
        "constant": lambda: torch.cat(
            [torch.randn(20, 2), torch.ones(20, 1), torch.randn(20, 2)], dim=1
        ),
        "zero_variance": lambda: torch.cat(
            [torch.randn(20, 2), torch.zeros(20, 1), torch.randn(20, 2)], dim=1
        ),
        "rank_deficient": lambda: torch.randn(20, 5)
        @ torch.randn(5, 3)
        @ torch.randn(3, 5),
    }

    if pattern not in patterns:
        raise ValueError(f"Unknown pattern: {pattern}")

    return patterns[pattern]()


def _add_missing(data, ratio):
    """Helper to add missing values to data."""
    missing_mask = torch.rand_like(data) < ratio
    data[missing_mask] = float("nan")
    return data


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================


@pytest.mark.basic
def test_basic_functionality(method, basic_data):
    """Test basic functionality of all PCA methods."""
    result = pca(basic_data, method=method, n_components=3)
    validate_pca_result(result, basic_data, method, 3)


# =============================================================================
# MISSING VALUES TESTS
# =============================================================================


@pytest.mark.missing_values
class TestMissingValues:
    """Tests for missing value handling."""

    def test_missing_values_handling(self, missing_method, missing_data):
        """Test PCA methods with missing values."""
        result = pca(missing_data, method=missing_method, n_components=2)
        validate_pca_result(result, missing_data, missing_method, 2)

    @pytest.mark.parametrize(
        "pattern", ["row_missing", "column_missing", "high_missing"]
    )
    def test_missing_patterns(self, pattern):
        """Test different missing data patterns."""
        data = create_special_data(pattern)
        # Use methods that handle missing values well
        missing_values_methods: list[AllowedMethod] = [
            "nipals",
            "ppca",
            "nlpca",
        ]
        for method in missing_values_methods:
            result = pca(data, method=method, n_components=2, max_iter=50)
            assert torch.isfinite(result.scores).all()


# =============================================================================
# EDGE CASES AND VALIDATION TESTS
# =============================================================================


@pytest.mark.edge_cases
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "n_samples,n_features,n_comp",
        [
            (10, 4, 2),  # Small data
            (20, 5, 1),  # Single component
            (10, 10, 5),  # Square data
            (5, 20, 3),  # Wide data
            (100, 5, 5),  # All components
        ],
    )
    def test_data_dimensions(self, n_samples, n_features, n_comp):
        """Test various data dimensions."""
        data = generate_test_data(n_samples=n_samples, n_features=n_features)
        result = pca(data, method="svd", n_components=n_comp)
        assert result.scores.shape == (n_samples, n_comp)

    @pytest.mark.parametrize(
        "pattern", ["constant", "zero_variance", "rank_deficient"]
    )
    def test_special_data_patterns(self, pattern):
        """Test with special data patterns."""
        data = create_special_data(pattern)
        # SVD should handle these gracefully
        result = pca(data, method="svd", n_components=2)
        validate_pca_result(result, data, "svd", 2)


@pytest.mark.validation
class TestValidation:
    """Tests for parameter and data validation."""

    @pytest.mark.parametrize(
        "kwargs,error_type",
        [
            ({"method": "invalid_method"}, ValueError),
            ({"method": "svd", "n_components": -1}, ValueError),
            ({"method": "svd", "n_components": 0}, ValueError),
        ],
    )
    def test_parameter_validation(self, kwargs, error_type):
        """Test parameter validation."""
        data = torch.randn(20, 5)
        with pytest.raises(error_type):
            pca(data, **kwargs)

    @pytest.mark.parametrize(
        "invalid_data",
        [
            torch.randn(10),  # 1D data
            torch.randn(10, 5, 3),  # 3D data
            torch.randn(1, 5),  # Too few samples
            torch.randn(10, 1),  # Too few features
            torch.randn(0, 5),  # No samples
            torch.randn(10, 0),  # No features
        ],
    )
    def test_invalid_data_shapes(self, invalid_data):
        """Test data shape validation."""
        with pytest.raises(ValueError):
            pca(invalid_data, method="svd")


# =============================================================================
# ADDITIONAL CRITICAL TESTS
# =============================================================================


@pytest.mark.validation
class TestAdditionalValidation:
    """Additional validation tests for missing edge cases."""

    def test_n_components_exceeds_dimensions(self):
        """Test when n_components exceeds min(n_samples, n_features)."""
        data = torch.randn(10, 5)
        # Should handle gracefully, not fail
        result = pca(data, method="svd", n_components=8)
        # Should cap at min(n_samples, n_features)
        assert result.scores.shape[1] <= min(data.shape)

    def test_tolerance_parameter_validation(self):
        """Test negative tolerance parameter validation."""
        data = torch.randn(20, 5)
        with pytest.raises(ValueError, match="tolerance must be positive"):
            pca(data, method="nipals", n_components=2, tolerance=-0.01)

    def test_max_iter_parameter_validation(self):
        """Test negative max_iter parameter validation."""
        data = torch.randn(20, 5)
        with pytest.raises(ValueError, match="max_iter must be positive"):
            pca(data, method="nipals", n_components=2, max_iter=-10)

    def test_all_nan_data(self):
        """Test behavior with completely NaN data."""
        data = torch.full((10, 5), float("nan"))
        missing_methods: list[AllowedMethod] = [
            "nipals",
            "ppca",
            "bpca",
            "svd_impute",
        ]
        for method in missing_methods:
            with pytest.raises((ValueError, RuntimeError)):
                pca(data, method=method, n_components=2)

    def test_single_valid_value_data(self):
        """Test data with only one non-NaN value."""
        data = torch.full((10, 5), float("nan"))
        data[0, 0] = 1.0
        missing_methods: list[AllowedMethod] = ["nipals", "ppca", "svd_impute"]
        for method in missing_methods:
            with pytest.raises((ValueError, RuntimeError)):
                pca(data, method=method, n_components=2)

    def test_infinite_values(self):
        """Test handling of infinite values."""
        data = torch.randn(20, 5)
        data[0, 0] = float("inf")
        data[1, 1] = float("-inf")

        # Most methods should fail gracefully or handle infinities
        with pytest.raises((ValueError, RuntimeError)):
            pca(data, method="svd", n_components=2)

    def test_non_finite_reconstruction(self):
        """Test reconstruction when result contains non-finite values."""
        # This tests the reconstruction validation
        data = generate_test_data(n_samples=20, n_features=5)
        result = pca(data, method="svd", n_components=3)

        # Test invalid n_components for reconstruction
        with pytest.raises((ValueError, IndexError)):
            result.reconstruct(n_components=10)  # More than available

    def test_empty_tensor_validation(self):
        """Test validation catches empty tensors properly."""
        # These should be caught by _check_data
        with pytest.raises(ValueError):
            pca(torch.empty(0, 5), method="svd")
        with pytest.raises(ValueError):
            pca(torch.empty(5, 0), method="svd")


# =============================================================================
# METHOD-SPECIFIC TESTS
# =============================================================================


@pytest.mark.method_specific
class TestMethodSpecific:
    """Tests for method-specific parameters and behavior."""

    @pytest.mark.parametrize(
        "method,params",
        [
            ("nipals", {"max_iter": 50, "tolerance": 1e-6}),
            ("ppca", {"max_iter": 100, "tolerance": 1e-5}),
            ("bpca", {"max_iter": 75, "tolerance": 1e-5}),
        ],
    )
    def test_iterative_parameters(self, method, params, missing_data):
        """Test iterative methods with custom parameters."""
        result = pca(missing_data, method=method, n_components=2, **params)
        validate_pca_result(result, missing_data, method, 2)

    def test_preprocessing_options(self, basic_data):
        """Test different preprocessing options."""
        # Test centering
        result_centered = pca(
            basic_data, method="svd", n_components=2, center=True, scale=False
        )
        result_uncentered = pca(
            basic_data, method="svd", n_components=2, center=False, scale=False
        )
        assert not torch.allclose(
            result_centered.scores, result_uncentered.scores
        )

        # Test scaling
        result_scaled = pca(
            basic_data, method="svd", n_components=2, center=True, scale=True
        )
        assert not torch.allclose(result_centered.scores, result_scaled.scores)


# =============================================================================
# DATA TYPE AND DEVICE TESTS
# =============================================================================


@pytest.mark.data_types
class TestDataTypes:
    """Tests for different data types and devices."""

    def test_double_precision(self):
        """Test with double precision."""
        data = torch.randn(20, 5, dtype=torch.float64)
        result = pca(data, method="svd", n_components=2)
        assert result.scores.dtype == torch.float64
        assert result.loadings.dtype == torch.float64
        assert result.eigenvalues.dtype == torch.float64

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test with CUDA device."""
        data = torch.randn(20, 5).cuda()
        result = pca(data, method="svd", n_components=2)
        assert result.scores.device.type == "cuda"
        assert result.loadings.device.type == "cuda"


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================


@pytest.mark.numerical
class TestNumericalStability:
    """Tests for numerical stability."""

    @pytest.mark.parametrize("scale_factor", [1e-10, 1e10])
    def test_extreme_scales(self, scale_factor):
        """Test with extreme value scales."""
        data = torch.randn(20, 5) * scale_factor
        result = pca(data, method="svd", n_components=2)
        assert torch.isfinite(result.scores).all()

    def test_mixed_scales(self):
        """Test with mixed feature scales."""
        data = torch.randn(30, 5)
        data[:, 0] *= 1e-6
        data[:, 1] *= 1e6
        data[:, 2] *= 1e-3
        data[:, 3] *= 1e3

        # Should handle with scaling
        result = pca(data, method="svd", n_components=3, scale=True)
        assert torch.isfinite(result.scores).all()

        # Eigenvalues should be reasonable
        assert (result.eigenvalues > 0).all()
        assert (result.eigenvalues < 1e10).all()


# =============================================================================
# VARIANCE EXPLAINED TESTS
# =============================================================================


@pytest.mark.variance
class TestVarianceExplained:
    """Tests for variance explained calculations."""

    def test_variance_properties(self, basic_data):
        """Test basic properties of explained variance."""
        result = pca(basic_data, method="svd", n_components=5)

        # Eigenvalues should be positive and decreasing
        assert (result.eigenvalues > 0).all()
        assert torch.all(result.eigenvalues[:-1] >= result.eigenvalues[1:])

        # Variance ratios should be valid probabilities
        ratios = result.explained_variance_ratio
        assert (ratios >= 0).all()
        assert (ratios <= 1).all()
        assert ratios.sum() <= 1.0 + 1e-6  # Allow small numerical error

    def test_variance_consistency(self, basic_data):
        """Test variance consistency across component counts."""
        results = [
            pca(basic_data, method="svd", n_components=n) for n in [2, 3, 4]
        ]

        # Eigenvalues should be consistent
        for i in range(len(results) - 1):
            n_comp = results[i].eigenvalues.shape[0]
            assert torch.allclose(
                results[i].eigenvalues,
                results[i + 1].eigenvalues[:n_comp],
                atol=1e-6,
            )


# =============================================================================
# COMPARISON TESTS
# =============================================================================


@pytest.mark.comparison
class TestComparisons:
    """Tests comparing different methods and implementations."""

    def test_svd_nipals_consistency(self):
        """Compare SVD and NIPALS on clean data."""
        data = generate_test_data(missing_ratio=0, noise_level=0.01)

        result_svd = pca(data, method="svd", n_components=2)
        result_nipals = pca(
            data, method="nipals", n_components=2, max_iter=500
        )

        # Account for sign ambiguity in PCA
        for i in range(2):
            if (
                torch.corrcoef(
                    torch.stack(
                        [result_svd.scores[:, i], result_nipals.scores[:, i]]
                    )
                )[0, 1]
                < 0
            ):
                result_nipals.scores[:, i] *= -1

        # Check similarity (allowing for some numerical differences)
        score_similarity = torch.corrcoef(
            torch.cat([result_svd.scores, result_nipals.scores], dim=1).T
        )
        # Diagonal blocks should have high correlation
        assert (score_similarity[:2, 2:4].diag() > 0.99).all()

    def test_sklearn_compatibility(self, basic_data):
        """Compare with scikit-learn PCA."""
        data_np = basic_data.numpy()
        n_comp = 3

        # Scikit-learn PCA
        sk_pca = PCA(n_components=n_comp, svd_solver="full")
        sk_pca.fit(data_np)

        # Our PCA
        result = pca(
            basic_data, method="svd", n_components=n_comp, center=True
        )

        # Compare explained variance
        sk_variance = torch.from_numpy(sk_pca.explained_variance_).float()
        torch_variance = result.eigenvalues

        relative_error = torch.abs(sk_variance - torch_variance) / sk_variance
        assert (relative_error < 0.01).all()  # Less than 1% error


# =============================================================================
# COMPONENT TESTS
# =============================================================================


@pytest.mark.components
class TestComponents:
    """Tests for PCA components properties."""

    def test_orthogonality(self, basic_data):
        """Test that components are orthogonal."""
        result = pca(basic_data, method="svd", n_components=3)

        # Components should be orthonormal
        gram = result.components @ result.components.T
        expected = torch.eye(3)
        assert torch.allclose(gram, expected, atol=1e-6)

        # For SVD, the scores are scaled by singular values
        # The covariance of scores should match eigenvalues
        scores_centered = result.scores - result.scores.mean(dim=0)
        n_samples = result.scores.shape[0]
        scores_cov = scores_centered.T @ scores_centered / (n_samples - 1)
        assert torch.allclose(
            scores_cov, torch.diag(result.eigenvalues), atol=1e-4
        )

    def test_reconstruction_error(self):
        """Test reconstruction error decreases with more components."""
        data = generate_test_data(missing_ratio=0, noise_level=0.01)
        errors = []

        for n_comp in [1, 2, 3, 4, 5]:
            result = pca(data, method="svd", n_components=n_comp)
            recon = result.reconstruct()
            error = torch.norm(data - recon) / torch.norm(data)
            errors.append(error)

        # Errors should decrease
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1]


# =============================================================================
# ROBUST PCA TESTS
# =============================================================================


@pytest.mark.robust
def test_robust_pca(outlier_data):
    """Test robust PCA with outliers."""
    # Standard PCA
    result_standard = pca(outlier_data, method="svd", n_components=2)

    # Robust PCA
    result_robust = pca(outlier_data, method="rpca", n_components=2)

    # Both should produce valid results
    validate_pca_result(result_standard, outlier_data, "svd", 2)
    validate_pca_result(result_robust, outlier_data, "rpca", 2)

    # Results should be different due to outlier handling
    assert not torch.allclose(
        result_standard.scores, result_robust.scores, atol=1e-3
    )


# =============================================================================
# RECONSTRUCTION TESTS
# =============================================================================


@pytest.mark.reconstruction
class TestReconstruction:
    """Tests for reconstruction functionality."""

    def test_reconstruction_shapes(self, basic_data):
        """Test reconstruction output shapes."""
        result = pca(basic_data, method="svd", n_components=5)

        # Test different numbers of components
        for n_comp in [1, 3, 5]:
            recon = result.reconstruct(n_components=n_comp)
            assert recon.shape == basic_data.shape
            assert torch.isfinite(recon).all()

    def test_reconstruction_quality(self):
        """Test reconstruction quality metrics."""
        data = generate_test_data(missing_ratio=0, noise_level=0.01)
        result = pca(data, method="svd", n_components=5)

        # Full reconstruction should be very close to original
        recon_full = result.reconstruct()
        rel_error = torch.norm(data - recon_full) / torch.norm(data)
        assert rel_error < 0.1  # Less than 10% error

        # Partial reconstruction should have higher error
        recon_partial = result.reconstruct(n_components=2)
        rel_error_partial = torch.norm(data - recon_partial) / torch.norm(data)
        assert rel_error_partial > rel_error


# =============================================================================
# NLPCA-SPECIFIC TESTS
# =============================================================================


@pytest.mark.nlpca
class TestNLPCA:
    """Tests specific to nonlinear PCA."""

    def test_basic_functionality(self):
        """Test NLPCA basic functionality."""
        data = generate_test_data(
            n_samples=50, n_features=10, noise_level=0.05
        )
        result = pca(
            data, method="nlpca", n_components=2, max_iter=30, verbose=False
        )
        validate_pca_result(result, data, "nlpca", 2)

    @pytest.mark.parametrize(
        "architecture",
        [
            [2, 5, 8],  # simple architecture
            [2, 10, 8],  # wider hidden layer
            [2, 4, 6, 8],  # deeper network with 4 layers
        ],
    )
    def test_architectures(self, architecture):
        """Test different network architectures."""
        data = generate_test_data(n_samples=50, n_features=8)

        result = pca(
            data,
            method="nlpca",
            n_components=2,
            units_per_layer=architecture,
            max_iter=20,
            verbose=False,
        )
        assert result.scores.shape == (50, 2)

    def test_regularization(self):
        """Test effect of weight decay."""
        data = generate_test_data(n_samples=50, n_features=8)

        # Different regularization strengths
        results = []
        for wd in [0.0, 0.01, 0.1]:
            result = pca(
                data,
                method="nlpca",
                n_components=2,
                weight_decay=wd,
                max_iter=30,
                verbose=False,
            )
            results.append(result)

        # Results should differ with different regularization
        assert not torch.allclose(
            results[0].scores, results[2].scores, atol=1e-3
        )

    def test_missing_values(self):
        """Test NLPCA with missing values."""
        data = generate_test_data(n_samples=50, missing_ratio=0.1)
        result = pca(
            data, method="nlpca", n_components=2, max_iter=30, verbose=False
        )
        assert torch.isfinite(result.scores).all()

    def test_network_storage(self):
        """Test network information storage."""
        data = generate_test_data(n_samples=30, n_features=8)
        result = pca(
            data, method="nlpca", n_components=2, max_iter=20, verbose=False
        )

        # Check network is stored
        assert hasattr(result, "net")
        assert result.net is not None

        # Test reconstruction works
        recon = result.reconstruct()
        assert recon.shape == data.shape
        assert torch.isfinite(recon).all()


# =============================================================================
# REPRODUCIBILITY TESTS
# =============================================================================


@pytest.mark.reproducibility
class TestReproducibility:
    """Tests for reproducibility of results."""

    @pytest.mark.parametrize("method", ["svd", "nipals", "ppca"])
    def test_deterministic_methods(self, method):
        """Test deterministic methods produce same results."""
        data = generate_test_data(seed=123)

        results = []
        for _ in range(3):
            torch.manual_seed(42)
            result = pca(data, method=method, n_components=2)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert torch.allclose(
                results[0].scores, results[i].scores, atol=1e-6
            )
            assert torch.allclose(
                results[0].eigenvalues, results[i].eigenvalues, atol=1e-6
            )


# =============================================================================
# PERFORMANCE TESTS (marked as slow)
# =============================================================================


@pytest.mark.slow
@pytest.mark.performance
class TestPerformance:
    """Performance tests with larger datasets."""

    @pytest.mark.parametrize(
        "n_samples,n_features,n_components",
        [
            (500, 100, 10),
            (100, 500, 10),
            (1000, 50, 5),
        ],
    )
    def test_large_datasets(self, n_samples, n_features, n_components):
        """Test with larger datasets."""
        data = torch.randn(n_samples, n_features)
        result = pca(data, method="svd", n_components=n_components)

        assert result.scores.shape == (n_samples, n_components)
        assert result.loadings.shape == (n_features, n_components)


# =============================================================================
# MISSING DATA PATTERN TESTS
# =============================================================================


@pytest.mark.missing_values
class TestAdditionalMissingPatterns:
    """Additional tests for complex missing data patterns."""

    def test_structured_missing_patterns(self):
        """Test structured missing data patterns."""
        # Block missing pattern
        data = torch.randn(20, 10)
        data[:10, :5] = float("nan")  # Top-left block missing

        result = pca(data, method="ppca", n_components=3, max_iter=50)
        assert torch.isfinite(result.scores).all()

    def test_random_sparse_missing(self):
        """Test very sparse data (high missing ratio)."""
        data = generate_test_data(n_samples=50, missing_ratio=0.9)

        # Should handle or fail gracefully
        try:
            result = pca(data, method="ppca", n_components=2, max_iter=100)
            assert torch.isfinite(result.scores).all()
        except (ValueError, RuntimeError):
            # Acceptable to fail with such high missingness
            pass

    def test_missing_entire_rows_and_columns(self):
        """Test data with entire rows and columns missing."""
        data = torch.randn(20, 10)
        data[5, :] = float("nan")  # Entire row missing
        data[:, 3] = float("nan")  # Entire column missing

        missing_methods: list[AllowedMethod] = ["nipals", "ppca", "bpca"]
        for method in missing_methods:
            # Should handle or provide meaningful error
            try:
                result = pca(data, method=method, n_components=2, max_iter=50)
                assert torch.isfinite(result.scores).all()
            except (ValueError, RuntimeError):
                # Acceptable behavior for extreme missingness
                pass


# =============================================================================
# CONVERGENCE AND STABILITY TESTS
# =============================================================================


@pytest.mark.numerical
class TestConvergenceStability:
    """Tests for convergence behavior and numerical stability."""

    @pytest.mark.parametrize("method", ["nipals", "rnipals", "ppca", "bpca"])
    def test_convergence_with_low_tolerance(self, method):
        """Test convergence with very strict tolerance."""
        data = generate_test_data(missing_ratio=0.1)

        # Very strict tolerance should still converge or hit max_iter
        result = pca(
            data, method=method, n_components=2, tolerance=1e-12, max_iter=1000
        )
        assert torch.isfinite(result.scores).all()

    @pytest.mark.parametrize("method", ["nipals", "ppca", "bpca"])
    def test_insufficient_iterations(self, method):
        """Test behavior with very few iterations."""
        data = generate_test_data(missing_ratio=0.2)

        # Very few iterations - should still produce results
        result = pca(
            data, method=method, n_components=2, max_iter=2  # Very low
        )
        assert torch.isfinite(result.scores).all()

    def test_nlpca_convergence_monitoring(self):
        """Test NLPCA convergence behavior."""
        data = generate_test_data(n_samples=40, n_features=8)

        # Test with verbose to ensure convergence monitoring works
        result = pca(
            data,
            method="nlpca",
            n_components=2,
            max_iter=50,
            verbose=True,  # Should not crash with verbose output
        )
        assert torch.isfinite(result.scores).all()


# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================


@pytest.mark.mathematical
class TestMathematicalProperties:
    """Tests for mathematical properties that should hold."""

    def test_eigenvalue_ordering_consistency(self):
        """Test that eigenvalues are consistently ordered across methods."""
        data = generate_test_data(missing_ratio=0.0, noise_level=0.01)

        methods_to_compare: list[AllowedMethod] = ["svd", "nipals"]
        results = {}

        for method in methods_to_compare:
            results[method] = pca(data, method=method, n_components=4)

        # Eigenvalues should be in descending order for all methods
        for method, result in results.items():
            eigenvals = result.eigenvalues
            assert torch.all(
                eigenvals[:-1] >= eigenvals[1:]
            ), f"Eigenvalues not ordered for {method}"

    def test_score_loading_relationship(self):
        """Test mathematical relationship between scores and loadings."""
        data = generate_test_data(missing_ratio=0.0)
        result = pca(
            data, method="svd", n_components=3, center=True, scale=False
        )

        # For SVD: data ≈ scores @ loadings.T (after centering)
        centered_data = data - data.mean(dim=0)
        reconstructed = result.scores @ result.loadings.T

        # Should be very close for clean SVD
        relative_error = torch.norm(
            centered_data - reconstructed
        ) / torch.norm(centered_data)
        assert relative_error < 0.1

    def test_explained_variance_sum_property(self):
        """Test that explained variance ratios sum to <= 1."""
        data = generate_test_data()

        for n_comp in [2, 3, 5]:
            result = pca(data, method="svd", n_components=n_comp)
            variance_sum = result.explained_variance_ratio.sum()
            assert variance_sum <= 1.0 + 1e-6  # Allow numerical precision
            assert (result.explained_variance_ratio >= 0).all()


# =============================================================================
# DTYPE AND PRECISION TESTS
# =============================================================================


@pytest.mark.data_types
class TestAdditionalDataTypes:
    """Additional tests for data types and precision."""

    def test_mixed_precision_consistency(self):
        """Test consistency between float32 and float64."""
        data_f32 = torch.randn(30, 8, dtype=torch.float32)
        data_f64 = data_f32.double()

        result_f32 = pca(data_f32, method="svd", n_components=3)
        result_f64 = pca(data_f64, method="svd", n_components=3)

        # Results should be similar (accounting for precision differences)
        score_diff = torch.norm(result_f32.scores.double() - result_f64.scores)
        total_norm = torch.norm(result_f64.scores)
        relative_diff = score_diff / total_norm

        assert relative_diff < 0.01  # Less than 1% difference

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_cpu_consistency(self):
        """Test consistency between CPU and CUDA computations."""
        data_cpu = torch.randn(30, 8)
        data_cuda = data_cpu.cuda()

        result_cpu = pca(data_cpu, method="svd", n_components=3)
        result_cuda = pca(data_cuda, method="svd", n_components=3)

        # Move CUDA results to CPU for comparisonscores_cuda_cpu
        scores_cuda_cpu = result_cuda.scores.cpu()
        loadings_cuda_cpu = result_cuda.loadings.cpu()

        # Handle sign ambiguity by aligning signs component by component
        for i in range(3):
            # Check correlation between CPU and CUDA scores for this component
            if torch.dot(result_cpu.scores[:, i], scores_cuda_cpu[:, i]) < 0:
                scores_cuda_cpu[:, i] *= -1
                loadings_cuda_cpu[:, i] *= -1

        # Now compare with appropriate tolerance
        score_diff = torch.norm(result_cpu.scores - scores_cuda_cpu)
        total_norm = torch.norm(result_cpu.scores)
        relative_diff = score_diff / total_norm

        assert (
            relative_diff < 1e-4
        )  # Relaxed tolerance for CPU/GPU differences

        # Also check loadings
        loadings_diff = torch.norm(result_cpu.loadings - loadings_cuda_cpu)
        loadings_norm = torch.norm(result_cpu.loadings)
        assert loadings_diff / loadings_norm < 1e-4

        # Eigenvalues should be very close (no sign ambiguity here)
        assert torch.allclose(
            result_cpu.eigenvalues, result_cuda.eigenvalues.cpu(), rtol=1e-4
        )


# =============================================================================
