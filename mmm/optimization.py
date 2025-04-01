import numpy as np
import pandas as pd
from scipy.optimize import minimize

def predict_sales(model, data, spend_values, channel_names):
    """
    Predict sales based on model and new spend values.
    
    This function applies the necessary transformations and predicts sales
    based on a new budget allocation.
    
    Parameters:
    -----------
    model : fitted model
        Trained model with coefficients
    data : pandas.DataFrame
        Original data with transformed features
    spend_values : dict
        Dictionary mapping channel names to new spend values
    channel_names : list
        List of original channel names
        
    Returns:
    --------
    float
        Predicted sales
    """
    # Create a copy of the data
    new_data = data.copy()
    
    # Update the spend values
    for channel in channel_names:
        if channel in spend_values:
            new_data[channel] = spend_values[channel]
    
    # TODO: Apply transformations (adstock, saturation) to the new data
    # For simplicity, we'll skip this step in this MVP
    
    # Predict sales
    X_cols = [col for col in data.columns if col not in ['Date', 'Sales', 'Noise']]
    X = new_data[X_cols]
    
    predicted_sales = model.predict(X).mean()
    
    return predicted_sales

def optimize_budget(
    model, 
    data, 
    channel_names, 
    total_budget,
    min_values=None,
    max_values=None,
    objective='sales'
):
    """
    Optimize marketing budget allocation to maximize sales or ROI.
    
    Parameters:
    -----------
    model : fitted model
        Trained model with coefficients
    data : pandas.DataFrame
        Data with transformed features
    channel_names : list
        List of channel names
    total_budget : float
        Total budget constraint
    min_values : dict, optional
        Minimum spend for each channel
    max_values : dict, optional
        Maximum spend for each channel
    objective : str, default='sales'
        Optimization objective ('sales' or 'roi')
        
    Returns:
    --------
    tuple
        (optimized allocation, expected sales)
    """
    # Set default min and max values if not provided
    if min_values is None:
        min_values = {channel: 0 for channel in channel_names}
    
    if max_values is None:
        # Default to current spend * 2
        max_values = {channel: data[channel].mean() * 2 for channel in channel_names}
    
    # Get current spend
    current_spend = {channel: data[channel].mean() for channel in channel_names}
    
    # Define the objective function (negative because we want to maximize)
    def objective_function(x):
        # Convert the flat array to a dictionary
        spend_values = {channel: x[i] for i, channel in enumerate(channel_names)}
        
        # Predict sales with the new allocation
        predicted_sales = predict_sales(model, data, spend_values, channel_names)
        
        if objective == 'sales':
            # Maximize sales
            return -predicted_sales
        elif objective == 'roi':
            # Maximize ROI (sales / total spend)
            return -predicted_sales / sum(x)
    
    # Define the constraint (total budget)
    def budget_constraint(x):
        return total_budget - sum(x)
    
    # Initial guess (current allocation)
    x0 = [current_spend[channel] for channel in channel_names]
    
    # Define bounds for each channel
    bounds = [(min_values[channel], max_values[channel]) for channel in channel_names]
    
    # Define constraints
    constraints = [{'type': 'eq', 'fun': budget_constraint}]
    
    # Perform optimization
    result = minimize(
        objective_function,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Extract the optimized allocation
    optimized_allocation = {
        channel: result.x[i] for i, channel in enumerate(channel_names)
    }
    
    # Calculate expected sales
    expected_sales = -objective_function(result.x)
    
    return optimized_allocation, expected_sales

def scenario_analysis(
    model, 
    data, 
    channel_names, 
    scenarios
):
    """
    Analyze different budget allocation scenarios.
    
    Parameters:
    -----------
    model : fitted model
        Trained model with coefficients
    data : pandas.DataFrame
        Data with transformed features
    channel_names : list
        List of channel names
    scenarios : dict
        Dictionary mapping scenario names to budget allocations
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with scenario analysis results
    """
    results = []
    
    for scenario_name, allocation in scenarios.items():
        # Ensure all channels are in the allocation
        for channel in channel_names:
            if channel not in allocation:
                allocation[channel] = data[channel].mean()
        
        # Predict sales
        predicted_sales = predict_sales(model, data, allocation, channel_names)
        
        # Calculate total spend
        total_spend = sum(allocation.values())
        
        # Calculate ROI
        roi = predicted_sales / total_spend
        
        results.append({
            'Scenario': scenario_name,
            'Predicted Sales': predicted_sales,
            'Total Spend': total_spend,
            'ROI': roi
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def budget_allocation_simulation(
    model, 
    data, 
    channel_names, 
    budget_range,
    steps=10
):
    """
    Simulate optimal budget allocations for different total budgets.
    
    Parameters:
    -----------
    model : fitted model
        Trained model with coefficients
    data : pandas.DataFrame
        Data with transformed features
    channel_names : list
        List of channel names
    budget_range : tuple
        (min_budget, max_budget) range to simulate
    steps : int, default=10
        Number of budget levels to simulate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with simulation results
    """
    min_budget, max_budget = budget_range
    
    # Create list of budgets to test
    budgets = np.linspace(min_budget, max_budget, steps)
    
    results = []
    
    for budget in budgets:
        # Optimize allocation for this budget
        allocation, sales = optimize_budget(
            model, data, channel_names, budget
        )
        
        # Calculate ROI
        roi = sales / budget
        
        # Add to results
        result = {
            'Total Budget': budget,
            'Predicted Sales': sales,
            'ROI': roi
        }
        
        # Add channel allocations
        for channel in channel_names:
            result[f'{channel} Allocation'] = allocation[channel]
            result[f'{channel} Share (%)'] = allocation[channel] / budget * 100
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

if __name__ == "__main__":
    # This module needs a trained model and data to be useful
    # See the Streamlit app for example usage
    pass