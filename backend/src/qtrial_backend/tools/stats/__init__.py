from __future__ import annotations

# Importing triggers @tool decorator registration
from qtrial_backend.tools.stats.column_stats import column_detailed_stats as column_detailed_stats
from qtrial_backend.tools.stats.value_counts import value_counts as value_counts
from qtrial_backend.tools.stats.correlation import correlation_matrix as correlation_matrix
from qtrial_backend.tools.stats.crosstab import cross_tabulation as cross_tabulation
from qtrial_backend.tools.stats.groupby import group_by_summary as group_by_summary
from qtrial_backend.tools.stats.missing import missing_data_patterns as missing_data_patterns
from qtrial_backend.tools.stats.distribution import distribution_info as distribution_info
from qtrial_backend.tools.stats.sample import sample_rows as sample_rows
from qtrial_backend.tools.stats.hypothesis_test import hypothesis_test as hypothesis_test
from qtrial_backend.tools.stats.survival import survival_analysis as survival_analysis
from qtrial_backend.tools.stats.outlier_detection import outlier_detection as outlier_detection
from qtrial_backend.tools.stats.normality_test import normality_test as normality_test
from qtrial_backend.tools.stats.pairwise_test import pairwise_group_test as pairwise_group_test
from qtrial_backend.tools.stats.effect_size import effect_size as effect_size
from qtrial_backend.tools.stats.duplicate_checks import duplicate_checks as duplicate_checks
from qtrial_backend.tools.stats.type_coercion import type_coercion_suggestions as type_coercion_suggestions
from qtrial_backend.tools.stats.baseline_balance import baseline_balance as baseline_balance
from qtrial_backend.tools.stats.regression import regression as regression
from qtrial_backend.tools.stats.multiple_testing import multiple_testing_correction as multiple_testing_correction
from qtrial_backend.tools.stats.plot_spec import plot_spec as plot_spec
from qtrial_backend.tools.stats.stat_test_selector import stat_test_selector as stat_test_selector
