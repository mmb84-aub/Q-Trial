"""Feature selection module with baseline methods and benchmarking framework."""

try:
    from qtrial_backend.feature_selection.mrmr import mrmr_selection
    from qtrial_backend.feature_selection.lasso import lasso_selection
    from qtrial_backend.feature_selection.univariate import univariate_selection
    from qtrial_backend.feature_selection.interface import select_features
    from qtrial_backend.feature_selection.benchmark import benchmark_all_methods, print_benchmark_summary, export_benchmark_results
except ImportError:
    # Allow partial imports
    pass

__all__ = [
    "mrmr_selection",
    "lasso_selection",
    "univariate_selection",
    "select_features",
    "benchmark_all_methods",
    "print_benchmark_summary",
    "export_benchmark_results",
]
