import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate common evaluation metrics for regression models.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Mean squared error
    mse = mean_squared_error(y_true, y_pred)
    
    # Root mean squared error
    rmse = np.sqrt(mse)
    
    # Mean absolute error
    mae = mean_absolute_error(y_true, y_pred)
    
    # R-squared
    r2 = r2_score(y_true, y_pred)
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Normalized root mean squared error
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'NRMSE': nrmse
    }

def decompose_contributions(model, X, original_channels=None, include_seasonality=True):
    """
    Decompose sales into channel contributions based on model coefficients.
    
    Parameters:
    -----------
    model : fitted model
        Trained model with coefficients
    X : pandas.DataFrame
        Feature matrix with transformed marketing variables
    original_channels : list, optional
        List of original channel names (before transformation)
    include_seasonality : bool, default=True
        Whether to include seasonality in the decomposition
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with decomposed contributions
    """
    # Get feature names
    if hasattr(X, 'columns'):
        feature_names = X.columns
    else:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Convert X to numpy array if it's a DataFrame
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # Get coefficients
    coefficients = model.coef_
    
    # Get intercept
    if hasattr(model, 'intercept_'):
        intercept = model.intercept_
    else:
        intercept = 0
    
    # Calculate contribution of each feature
    contributions = {}
    contributions['Intercept'] = np.ones(X_array.shape[0]) * intercept
    
    for i, feature in enumerate(feature_names):
        contributions[feature] = X_array[:, i] * coefficients[i]
    
    # Convert to DataFrame
    contributions_df = pd.DataFrame(contributions)
    
    # Map transformed features back to original channels if provided
    if original_channels is not None:
        # Create a mapping from transformed features to original channels
        feature_to_channel = {}
        for channel in original_channels:
            for feature in feature_names:
                if channel in feature:
                    feature_to_channel[feature] = channel
        
        # Group contributions by original channel
        channel_contributions = {}
        channel_contributions['Intercept'] = contributions_df['Intercept']
        
        for channel in original_channels:
            channel_features = [f for f in feature_names if channel in f]
            if channel_features:
                channel_contributions[channel] = contributions_df[channel_features].sum(axis=1)
            else:
                channel_contributions[channel] = np.zeros(X_array.shape[0])
        
        # Include seasonality if present and requested
        if include_seasonality and 'Seasonality' in feature_names:
            channel_contributions['Seasonality'] = contributions_df['Seasonality']
        
        # Include trend if present
        if 'Trend' in feature_names:
            channel_contributions['Trend'] = contributions_df['Trend']
        
        # Convert to DataFrame
        contributions_df = pd.DataFrame(channel_contributions)
    
    return contributions_df

def calculate_roi(model, X, spend_data, channel_cols, transformation_info=None):
    """
    Calculate ROI for each marketing channel.
    
    Parameters:
    -----------
    model : fitted model
        Trained model with coefficients
    X : pandas.DataFrame
        Feature matrix with transformed marketing variables
    spend_data : pandas.DataFrame
        Original marketing spend data
    channel_cols : list
        List of channel column names
    transformation_info : dict, optional
        Information about transformations applied
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with ROI calculations
    """
    # Get contributions
    contributions = decompose_contributions(model, X, original_channels=channel_cols)
    
    # Calculate total contribution by channel
    channel_total_contrib = {}
    for channel in channel_cols:
        if channel in contributions.columns:
            channel_total_contrib[channel] = contributions[channel].sum()
        else:
            channel_total_contrib[channel] = 0
    
    # Calculate total spend by channel
    channel_total_spend = {}
    for channel in channel_cols:
        if channel in spend_data.columns:
            channel_total_spend[channel] = spend_data[channel].sum()
        else:
            channel_total_spend[channel] = 1  # Avoid division by zero
    
    # Calculate ROI
    roi_data = []
    for channel in channel_cols:
        roi = channel_total_contrib[channel] / channel_total_spend[channel]
        
        roi_data.append({
            'Channel': channel,
            'Total Contribution': channel_total_contrib[channel],
            'Total Spend': channel_total_spend[channel],
            'ROI': roi
        })
    
    # Convert to DataFrame
    roi_df = pd.DataFrame(roi_data)
    
    return roi_df

def calculate_elasticity(model, X, channel_cols, transformation_info=None):
    """
    Calculate elasticity for each marketing channel.
    
    Elasticity measures the percentage change in sales for a percentage change in marketing spend.
    
    Parameters:
    -----------
    model : fitted model
        Trained model with coefficients
    X : pandas.DataFrame
        Feature matrix with transformed marketing variables
    channel_cols : list
        List of channel column names
    transformation_info : dict, optional
        Information about transformations applied
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with elasticity calculations
    """
    # This is a simplified elasticity calculation
    # In a real MMM, elasticity would account for transformations and diminishing returns
    
    # Get feature names
    if hasattr(X, 'columns'):
        feature_names = X.columns
    else:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Get coefficients
    coefficients = model.coef_
    
    # Calculate average input values
    X_mean = X.mean(axis=0)
    
    # Calculate predicted value at mean
    y_mean = model.predict(X_mean.values.reshape(1, -1))[0]
    
    # Calculate elasticity for each feature
    elasticity_data = []
    
    for channel in channel_cols:
        # Find features associated with this channel
        channel_features = [i for i, feature in enumerate(feature_names) if channel in feature]
        
        if channel_features:
            # Simple elasticity calculation for demonstration
            # In practice, this would account for transformations
            feature_idx = channel_features[0]
            elasticity = coefficients[feature_idx] * X_mean[feature_idx] / y_mean
            
            elasticity_data.append({
                'Channel': channel,
                'Elasticity': elasticity
            })
    
    # Convert to DataFrame
    elasticity_df = pd.DataFrame(elasticity_data)
    
    return elasticity_df

if __name__ == "__main__":
    # Example usage
    # Generate some dummy data
    n_samples = 100
    n_features = 3
    
    X = np.random.rand(n_samples, n_features)
    y_true = 2 + 3 * X[:, 0] + 1.5 * X[:, 1] - 2 * X[:, 2] + 0.5 * np.random.randn(n_samples)
    
    # Make predictions with some error
    y_pred = y_true + np.random.randn(n_samples) * 0.5
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print("Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")