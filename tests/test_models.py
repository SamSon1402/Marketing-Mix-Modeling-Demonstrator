"""
Unit tests for the models module.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmm.models import (
    train_linear_model,
    train_ridge_model,
    train_lasso_model,
    train_elasticnet_model,
    compare_models,
    prepare_model_comparison
)

class TestModels(unittest.TestCase):
    """Test cases for the models module."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)  # For reproducibility
        
        # Generate synthetic data for testing
        n_samples = 100
        n_features = 5
        
        self.X = np.random.rand(n_samples, n_features)
        self.y = (2 + 3 * self.X[:, 0] + 1.5 * self.X[:, 1] - 
                 2 * self.X[:, 2] + 0.5 * np.random.randn(n_samples))
        
        # Convert to pandas DataFrame for more realistic testing
        self.X_df = pd.DataFrame(
            self.X, 
            columns=[f'Feature_{i}' for i in range(n_features)]
        )
        self.y_series = pd.Series(self.y, name='Target')
    
    def test_train_linear_model(self):
        """Test the train_linear_model function."""
        # Test with numpy arrays
        model, predictions = train_linear_model(self.X, self.y)
        
        # Check model type
        self.assertIsInstance(model, LinearRegression)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(self.y))
        
        # Check that model can predict on new data
        new_X = np.random.rand(10, self.X.shape[1])
        new_predictions = model.predict(new_X)
        self.assertEqual(len(new_predictions), 10)
        
        # Test with pandas DataFrame/Series
        model_df, predictions_df = train_linear_model(self.X_df, self.y_series)
        
        # Check model type
        self.assertIsInstance(model_df, LinearRegression)
        
        # Check predictions shape
        self.assertEqual(len(predictions_df), len(self.y_series))
    
    def test_train_ridge_model(self):
        """Test the train_ridge_model function."""
        alpha = 1.0
        
        # Test with numpy arrays
        model, predictions = train_ridge_model(self.X, self.y, alpha=alpha)
        
        # Check model type
        self.assertIsInstance(model, Ridge)
        
        # Check alpha value
        self.assertEqual(model.alpha, alpha)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(self.y))
        
        # Test with different alpha
        alpha2 = 2.0
        model2, _ = train_ridge_model(self.X, self.y, alpha=alpha2)
        self.assertEqual(model2.alpha, alpha2)
    
    def test_train_lasso_model(self):
        """Test the train_lasso_model function."""
        alpha = 1.0
        
        # Test with numpy arrays
        model, predictions = train_lasso_model(self.X, self.y, alpha=alpha)
        
        # Check model type
        self.assertIsInstance(model, Lasso)
        
        # Check alpha value
        self.assertEqual(model.alpha, alpha)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(self.y))
    
    def test_train_elasticnet_model(self):
        """Test the train_elasticnet_model function."""
        alpha = 1.0
        l1_ratio = 0.5
        
        # Test with numpy arrays
        model, predictions = train_elasticnet_model(
            self.X, self.y, alpha=alpha, l1_ratio=l1_ratio
        )
        
        # Check model type
        self.assertIsInstance(model, ElasticNet)
        
        # Check parameters
        self.assertEqual(model.alpha, alpha)
        self.assertEqual(model.l1_ratio, l1_ratio)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(self.y))
    
    def test_compare_models(self):
        """Test the compare_models function."""
        # Prepare models
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1)
        }
        
        # Compare models
        results = compare_models(self.X, self.y, models, cv=3)
        
        # Check results
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), len(models))
        
        # Check column names
        expected_cols = ['mean_r2', 'std_r2', 'min_r2', 'max_r2']
        for col in expected_cols:
            self.assertIn(col, results.columns)
    
    def test_prepare_model_comparison(self):
        """Test the prepare_model_comparison function."""
        alphas = [0.01, 0.1, 1.0]
        
        # Get models
        models = prepare_model_comparison(self.X, self.y, alphas=alphas)
        
        # Check number of models
        # Linear + 3 Ridge + 3 Lasso + 3*3 ElasticNet
        expected_count = 1 + len(alphas) + len(alphas) + len(alphas) * 3
        self.assertEqual(len(models), expected_count)
        
        # Check model types
        model_types = set(type(model) for model in models.values())
        self.assertEqual(len(model_types), 3)  # LinearRegression, Ridge, Lasso, ElasticNet
        
        # Check model names
        self.assertIn('Linear Regression', models.keys())
        
        for alpha in alphas:
            self.assertIn(f'Ridge (alpha={alpha})', models.keys())
            self.assertIn(f'Lasso (alpha={alpha})', models.keys())

if __name__ == '__main__':
    unittest.main()