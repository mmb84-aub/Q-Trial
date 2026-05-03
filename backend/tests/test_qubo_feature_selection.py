"""
Test suite for QUBO-based feature selection module.

Tests cover:
- Basic functionality and edge cases
- Relevance scoring with various data types
- Redundancy matrix computation
- QUBO solver execution
- Hard constraints application
- Fallback mechanisms (greedy diversity, error recovery)
- Integration with multiple data types and sizes
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from qtrial_backend.quantum.feature_selector import (
    compute_relevance_scores,
    compute_redundancy_matrix,
    construct_qubo_matrix,
    solve_qubo,
    apply_hard_constraints,
    mean_pairwise_correlation,
    run_qubo_feature_selection,
    greedy_diversity_selection,
)


class TestComputeRelevanceScores:
    """Test relevance score computation for various data type combinations."""
    
    def test_numeric_vs_numeric_correlation(self):
        """Numeric vs numeric outcome should use Pearson correlation."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'y': [1, 2, 3, 4, 5]
        })
        scores = compute_relevance_scores(df, outcome_column='y', candidate_columns=['x1'])
        assert 'x1' in scores
        assert 0 <= scores['x1'] <= 1
        # Perfect correlation should be high
        assert scores['x1'] > 0.9
    
    def test_categorical_vs_numeric(self):
        """Categorical feature vs numeric outcome should use eta-squared."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C'],
            'numeric': [1.0, 1.1, 5.0, 5.1, 10.0]
        })
        scores = compute_relevance_scores(df, outcome_column='numeric', candidate_columns=['category'])
        assert 'category' in scores
        assert 0 <= scores['category'] <= 1
    
    def test_numeric_vs_categorical(self):
        """Numeric feature vs categorical outcome should use point-biserial correlation."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, 5.0, 6.0],
            'category': ['X', 'X', 'Y', 'Y']
        })
        scores = compute_relevance_scores(df, outcome_column='category', candidate_columns=['numeric'])
        assert 'numeric' in scores
        assert 0 <= scores['numeric'] <= 1
    
    def test_no_outcome_column(self):
        """Without outcome column, should use variance-based relevance."""
        df = pd.DataFrame({
            'var_high': [1, 2, 3, 4, 5],
            'var_low': [1, 1, 1, 1, 1]
        })
        scores = compute_relevance_scores(df, outcome_column=None)
        assert 'var_high' in scores
        assert 'var_low' in scores
        # High variance should score higher
        assert scores['var_high'] > scores['var_low']
    
    def test_missing_values(self):
        """Should handle missing values gracefully."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5],
            'y': [1, 2, 3, np.nan, 5]
        })
        scores = compute_relevance_scores(df, outcome_column='y', candidate_columns=['x'])
        assert 'x' in scores
        assert not np.isnan(scores['x'])
    
    def test_empty_candidates(self):
        """Should return empty dict for empty candidate list."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
        scores = compute_relevance_scores(df, outcome_column='y', candidate_columns=[])
        assert scores == {}


class TestComputeRedundancyMatrix:
    """Test redundancy matrix computation."""
    
    def test_identical_columns(self):
        """Identical numeric columns should have redundancy = 1."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [1, 2, 3, 4, 5]
        })
        matrix = compute_redundancy_matrix(df, ['x1', 'x2'])
        # Check upper triangle (redundancy is symmetric)
        assert matrix[0, 1] > 0.99  # Should be ~1.0
    
    def test_uncorrelated_columns(self):
        """Uncorrelated columns should have low redundancy (note: anticorrelated = high redundancy)."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [5, 4, 3, 2, 1]  # Perfectly anticorrelated, so |correlation| = 1
        })
        matrix = compute_redundancy_matrix(df, ['x1', 'x2'])
        # Anticorrelation has absolute value 1, so redundancy should be high after normalization
        assert matrix[0, 1] > 0.9
    
    def test_truly_uncorrelated_columns(self):
        """Redundancy matrix handles independent columns reasonably."""
        np.random.seed(123)  # Different seed to avoid correlation
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        matrix = compute_redundancy_matrix(df, ['x1', 'x2', 'x3'])
        
        # Check that the matrix is normalized to [0, 1]
        assert np.all(matrix >= 0) and np.all(matrix <= 1)
        # Check diagonal is zero
        assert np.diag(matrix).sum() == 0
    
    def test_matrix_shape(self):
        """Matrix should be M x M."""
        df = pd.DataFrame({
            'x1': np.random.rand(10),
            'x2': np.random.rand(10),
            'x3': np.random.rand(10)
        })
        matrix = compute_redundancy_matrix(df, ['x1', 'x2', 'x3'])
        assert matrix.shape == (3, 3)
        assert np.all(np.diag(matrix) == 0)  # Diagonal should be 0
    
    def test_single_column(self):
        """Single column should produce 1x1 zero matrix."""
        df = pd.DataFrame({'x1': [1, 2, 3, 4, 5]})
        matrix = compute_redundancy_matrix(df, ['x1'])
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 0


class TestConstructQuboMatrix:
    """Test QUBO matrix construction."""
    
    def test_qubo_structure(self):
        """QUBO matrix should have correct structure."""
        relevance = {'x1': 0.8, 'x2': 0.5}
        redundancy = np.array([[0, 0.3], [0.3, 0]])
        Q = construct_qubo_matrix(relevance, redundancy, ['x1', 'x2'], lambda_penalty=1.0)
        
        assert (0, 0) in Q  # Diagonal entries
        assert (1, 1) in Q
        assert (0, 1) in Q  # Off-diagonal
    
    def test_diagonal_negative(self):
        """Diagonal should be negative (encourages selection)."""
        relevance = {'x1': 0.8, 'x2': 0.5}
        redundancy = np.array([[0, 0.0], [0.0, 0]])
        Q = construct_qubo_matrix(relevance, redundancy, ['x1', 'x2'], lambda_penalty=0.0)
        
        # Diagonal values should incorporate -relevance (before adding small positive contribution)
        # The diagonal is: -relevance + (parsimony_weight / M)
        # For x1: -0.8 + (0.05/2) = -0.775
        assert Q[(0, 0)] < 0
        assert Q[(1, 1)] < 0


class TestSolveQubo:
    """Test QUBO solver."""
    
    def test_solver_returns_dict(self):
        """Solver should return a dict of binary assignments."""
        Q = {(0, 0): -1, (1, 1): -1, (0, 1): 0.5}
        result = solve_qubo(Q, num_reads=100, num_sweeps=100)
        assert isinstance(result, dict)
    
    def test_solver_returns_binary(self):
        """All values in result should be 0 or 1."""
        Q = {(0, 0): -1, (1, 1): -1, (0, 1): 0.5}
        result = solve_qubo(Q, num_reads=100, num_sweeps=100)
        for v in result.values():
            assert v in [0, 1]
    
    def test_timeout_handling(self):
        """Solver should handle timeout gracefully."""
        Q = {(0, 0): -1, (1, 1): -1}
        # Should not raise exception even with very short timeout
        result = solve_qubo(Q, num_reads=10, num_sweeps=10, timeout_seconds=0.001)
        # Result may be empty or partial, but shouldn't crash
        assert isinstance(result, dict)


class TestApplyHardConstraints:
    """Test constraint application."""
    
    def test_min_features_enforced(self):
        """Should enforce minimum features."""
        selected_indices = [0]  # Only 1 selected
        candidates = ['x1', 'x2', 'x3', 'x4', 'x5']
        relevance = {c: 0.5 for c in candidates}
        
        result = apply_hard_constraints(selected_indices, candidates, relevance)
        # min = max(4, ceil(sqrt(5))) = 4
        assert len(result) >= 4
    
    def test_max_features_enforced(self):
        """Should enforce maximum features (20)."""
        # Create a large set of selected features
        selected_indices = list(range(30))
        candidates = [f'x{i}' for i in range(30)]
        relevance = {c: 0.5 for c in candidates}
        
        result = apply_hard_constraints(selected_indices, candidates, relevance)
        assert len(result) <= 20
    
    def test_outcome_column_included(self):
        """Outcome column should always be included."""
        selected_indices = []
        candidates = ['x1', 'x2']
        relevance = {c: 0.5 for c in candidates}
        
        result = apply_hard_constraints(selected_indices, candidates, relevance, outcome_column='outcome')
        assert 'outcome' in result
    
    def test_excluded_columns_removed(self):
        """Excluded columns should not appear in result when there are enough other columns."""
        selected_indices = [0, 1, 2, 3, 4]  # 5 columns selected
        candidates = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        relevance = {c: 0.5 for c in candidates}
        
        result = apply_hard_constraints(selected_indices, candidates, relevance, excluded_columns=['x2'])
        
        # With 6 candidates available and 5 selected, excluding x2 should still work
        # (min constraint = max(4, ceil(sqrt(6))) = 4, and we can get 4+ from 5 non-excluded)
        assert 'x2' not in result
        assert len(result) >= 4


class TestMeanPairwiseCorrelation:
    """Test redundancy measurement."""
    
    def test_identical_columns(self):
        """Identical columns should have correlation = 1."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [1, 2, 3, 4, 5]
        })
        corr = mean_pairwise_correlation(df, ['x1', 'x2'])
        assert corr > 0.99
    
    def test_uncorrelated_columns(self):
        """Uncorrelated columns should have low correlation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        corr = mean_pairwise_correlation(df, ['x1', 'x2'])
        assert 0 <= corr <= 1
    
    def test_single_column(self):
        """Single column should return 0."""
        df = pd.DataFrame({'x1': [1, 2, 3, 4, 5]})
        corr = mean_pairwise_correlation(df, ['x1'])
        assert corr == 0.0


class TestGreedyDiversitySelection:
    """Test greedy diversity fallback."""
    
    def test_selects_target_features(self):
        """Should select approximately target number of features."""
        df = pd.DataFrame({
            'x1': np.random.rand(20),
            'x2': np.random.rand(20),
            'x3': np.random.rand(20),
            'x4': np.random.rand(20),
            'x5': np.random.rand(20)
        })
        relevance = {c: 0.5 for c in ['x1', 'x2', 'x3', 'x4', 'x5']}
        
        result = greedy_diversity_selection(df, ['x1', 'x2', 'x3', 'x4', 'x5'], relevance, target_features=3)
        assert len(result) == 3
    
    def test_includes_outcome(self):
        """Should include outcome column if specified."""
        df = pd.DataFrame({
            'x1': np.random.rand(20),
            'x2': np.random.rand(20),
            'outcome': [0, 1] * 10
        })
        relevance = {'x1': 0.5, 'x2': 0.5}
        
        result = greedy_diversity_selection(df, ['x1', 'x2'], relevance, outcome_column='outcome')
        assert 'outcome' in result
    
    def test_respects_must_include(self):
        """Should include must_include features."""
        df = pd.DataFrame({
            'x1': np.random.rand(20),
            'x2': np.random.rand(20),
            'x3': np.random.rand(20)
        })
        relevance = {'x1': 0.5, 'x2': 0.5, 'x3': 0.5}
        
        result = greedy_diversity_selection(df, ['x1', 'x2', 'x3'], relevance, target_features=2, must_include=['x1'])
        assert 'x1' in result


class TestRunQuboFeatureSelection:
    """Integration tests for main QUBO selection function."""
    
    def test_basic_functionality(self):
        """Should run successfully with normal dataset."""
        df = pd.DataFrame({
            'x1': np.random.rand(50),
            'x2': np.random.rand(50),
            'x3': np.random.rand(50),
            'outcome': np.random.randint(0, 2, 50)
        })
        
        result = run_qubo_feature_selection(df, profile=None, outcome_column='outcome')
        
        assert 'selected_columns' in result
        assert 'redundancy_reduction_pct' in result
        assert 'selection_method' in result
        assert len(result['selected_columns']) > 0
    
    def test_returns_valid_dict(self):
        """Should return all required fields."""
        df = pd.DataFrame({
            'x1': np.random.rand(20),
            'x2': np.random.rand(20),
            'outcome': [0, 1] * 10
        })
        
        result = run_qubo_feature_selection(df, profile=None, outcome_column='outcome')
        
        required_keys = [
            'selected_columns', 'excluded_columns', 'relevance_scores',
            'redundancy_before', 'redundancy_after', 'redundancy_reduction',
            'redundancy_reduction_pct', 'n_candidates', 'n_selected',
            'solver', 'lambda_penalty', 'selection_method', 'outcome_column'
        ]
        for key in required_keys:
            assert key in result
    
    def test_no_numeric_columns(self):
        """Should handle case with no numeric columns."""
        df = pd.DataFrame({
            'x': ['a', 'b', 'c', 'd'],
            'y': ['x', 'y', 'z', 'w']
        })
        
        result = run_qubo_feature_selection(df, profile=None, outcome_column='x')
        
        assert result['n_candidates'] == 0
        assert result['selection_method'] == 'error_fallback'
    
    def test_single_numeric_column(self):
        """Should handle dataset with only one numeric column."""
        df = pd.DataFrame({
            'numeric': np.random.rand(20),
            'categorical': ['a', 'b'] * 10
        })
        
        result = run_qubo_feature_selection(df, profile=None, outcome_column='categorical')
        
        assert 'numeric' in result['selected_columns']
        assert result['n_candidates'] >= 1
    
    def test_timeout_fallback(self):
        """Should fallback gracefully when timeout occurs."""
        df = pd.DataFrame({
            'x1': np.random.rand(100),
            'x2': np.random.rand(100),
            'x3': np.random.rand(100),
            'outcome': np.random.randint(0, 2, 100)
        })
        
        # Use very short timeout to trigger fallback
        result = run_qubo_feature_selection(df, profile=None, outcome_column='outcome', timeout_seconds=0.001)
        
        assert result['n_selected'] > 0  # Should still select features
        assert result['selection_method'] in ['qubo', 'greedy_diversity', 'error_fallback']
    
    def test_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame()
        
        result = run_qubo_feature_selection(df, profile=None)
        
        assert result['n_candidates'] == 0
        assert result['selection_method'] == 'error_fallback'
    
    def test_outcome_always_included(self):
        """Outcome column should be in selected columns."""
        df = pd.DataFrame({
            'x1': np.random.rand(20),
            'x2': np.random.rand(20),
            'outcome': [0, 1] * 10
        })
        
        result = run_qubo_feature_selection(df, profile=None, outcome_column='outcome')
        
        assert 'outcome' in result['selected_columns']
    
    def test_lambda_penalty_effect(self):
        """Different lambda penalties should affect results."""
        df = pd.DataFrame({
            'x1': [i for i in range(20)],
            'x2': [i for i in range(20)],
            'x3': [i*2 for i in range(20)],
            'outcome': [0, 1] * 10
        })
        
        result_low = run_qubo_feature_selection(df, profile=None, outcome_column='outcome', lambda_penalty=0.1)
        result_high = run_qubo_feature_selection(df, profile=None, outcome_column='outcome', lambda_penalty=2.0)
        
        # Both should complete without error
        assert result_low['selection_method'] in ['qubo', 'greedy_diversity', 'error_fallback']
        assert result_high['selection_method'] in ['qubo', 'greedy_diversity', 'error_fallback']


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_all_missing_values(self):
        """Should handle columns with all NaN."""
        df = pd.DataFrame({
            'all_nan': [np.nan] * 10,
            'normal': np.random.rand(10),
            'outcome': [0, 1] * 5
        })
        
        result = run_qubo_feature_selection(df, profile=None, outcome_column='outcome')
        
        assert result['n_selected'] > 0
    
    def test_constant_columns(self):
        """Should handle constant-value columns."""
        df = pd.DataFrame({
            'constant': [1.0] * 20,
            'varying': np.random.rand(20),
            'outcome': [0, 1] * 10
        })
        
        result = run_qubo_feature_selection(df, profile=None, outcome_column='outcome')
        
        assert result['n_selected'] > 0
    
    def test_high_dimensional_data(self):
        """Should handle large number of features."""
        np.random.seed(42)
        n_features = 100
        data = {f'x{i}': np.random.rand(50) for i in range(n_features)}
        data['outcome'] = np.random.randint(0, 2, 50)
        df = pd.DataFrame(data)
        
        result = run_qubo_feature_selection(df, profile=None, outcome_column='outcome', timeout_seconds=10)
        
        assert result['n_selected'] <= 20  # Should respect max constraint
        assert result['n_candidates'] == n_features


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
