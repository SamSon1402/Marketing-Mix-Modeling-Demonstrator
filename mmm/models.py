import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

def train_linear_model(X, y, test_size=0.2, random_state=42):
    """
    Train a linear regression model for marketing mix modeling.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix with transformed marketing variables
    y : pandas.Series or numpy.ndarray
        Target variable (usually sales)
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (trained model, predictions on full dataset)
    """
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get predictions on the full dataset
    predictions = model.predict(X)
    
    return model, predictions

def train_ridge_model(X, y, alpha=1.0, test_size=0.2, random_state=42):
    """
    Train a ridge regression model for marketing mix modeling.
    
    Ridge regression adds L2 regularization to prevent overfitting.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix with transformed marketing variables
    y : pandas.Series or numpy.ndarray
        Target variable (usually sales)
    alpha : float, default=1.0
        Regularization strength
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (trained model, predictions on full dataset)
    """
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train model
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Get predictions on the full dataset
    predictions = model.predict(X)
    
    return model, predictions

def train_lasso_model(X, y, alpha=1.0, test_size=0.2, random_state=42):
    """
    Train a lasso regression model for marketing mix modeling.
    
    Lasso regression adds L1 regularization for feature selection.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix with transformed marketing variables
    y : pandas.Series or numpy.ndarray
        Target variable (usually sales)
    alpha : float, default=1.0
        Regularization strength
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (trained model, predictions on full dataset)
    """
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train model
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Get predictions on the full dataset
    predictions = model.predict(X)
    
    return model, predictions

def train_elasticnet_model(X, y, alpha=1.0, l1_ratio=0.5, test_size=0.2, random_state=42):
    """
    Train an elastic net regression model for marketing mix modeling.
    
    Elastic Net combines L1 and L2 regularization.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix with transformed marketing variables
    y : pandas.Series or numpy.ndarray
        Target variable (usually sales)
    alpha : float, default=1.0
        Regularization strength
    l1_ratio : float, default=0.5
        Ratio of L1 regularization (0 = Ridge, 1 = Lasso)
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (trained model, predictions on full dataset)
    """
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)
    
    # Get predictions on the full dataset
    predictions = model.predict(X)
    
    return model, predictions

def compare_models(X, y, models_dict, cv=5):
    """
    Compare multiple models using cross-validation.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix with transformed marketing variables
    y : pandas.Series or numpy.ndarray
        Target variable (usually sales)
    models_dict : dict
        Dictionary mapping model names to initialized models
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cross-validation results for each model
    """
    results = {}
    
    for name, model in models_dict.items():
        # Get cross-validation scores
        scores = cross_val_score(
            model, X, y, cv=cv, scoring='r2'
        )
        
        results[name] = {
            'mean_r2': scores.mean(),
            'std_r2': scores.std(),
            'min_r2': scores.min(),
            'max_r2': scores.max()
        }
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results).T
    
    return results_df

def prepare_model_comparison(X, y, alphas=[0.01, 0.1, 1.0, 10.0]):
    """
    Prepare a comparison of different models with different hyperparameters.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix with transformed marketing variables
    y : pandas.Series or numpy.ndarray
        Target variable (usually sales)
    alphas : list, default=[0.01, 0.1, 1.0, 10.0]
        List of alpha values to try for regularized models
        
    Returns:
    --------
    dict
        Dictionary of initialized models with different configurations
    """
    models = {
        'Linear Regression': LinearRegression()
    }
    
    # Add Ridge models with different alphas
    for alpha in alphas:
        models[f'Ridge (alpha={alpha})'] = Ridge(alpha=alpha)
    
    # Add Lasso models with different alphas
    for alpha in alphas:
        models[f'Lasso (alpha={alpha})'] = Lasso(alpha=alpha)
    
    # Add ElasticNet models with different configurations
    for alpha in alphas:
        for l1_ratio in [0.2, 0.5, 0.8]:
            models[f'ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})'] = ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio
            )
    
    return models

if __name__ == "__main__":
    # Example usage
    # Generate some dummy data
    n_samples = 100
    n_features = 5
    
    X = np.random.rand(n_samples, n_features)
    y = 2 + 3 * X[:, 0] + 1.5 * X[:, 1] - 2 * X[:, 2] + 0.5 * np.random.randn(n_samples)
    
    # Train a linear model
    model, predictions = train_linear_model(X, y)
    
    # Print coefficients
    print("Linear Model Coefficients:")
    print(model.coef_)
    
    # Train a ridge model
    ridge_model, ridge_predictions = train_ridge_model(X, y, alpha=1.0)
    
    # Print coefficients
    print("\nRidge Model Coefficients:")
    print(ridge_model.coef_)