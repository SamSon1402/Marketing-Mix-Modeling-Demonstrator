import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mmm.transformations import (
    apply_adstock, 
    apply_saturation, 
    apply_hill_adstock,
    apply_s_curve,
    transform_marketing_data
)
from mmm.models import (
    train_linear_model,
    train_ridge_model,
    train_lasso_model,
    train_elasticnet_model,
    compare_models,
    prepare_model_comparison
)
from mmm.evaluation import calculate_metrics
from app.components.visualizations import plot_model_coefficients, plot_actual_vs_predicted

def render_model_training():
    """
    Render the model training page.
    """
    st.title("Model Training")
    
    # Check if data exists in session state
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("No data available. Please generate or upload data first.")
        return
    
    data = st.session_state.data
    channel_names = [col for col in data.columns if col not in ['Date', 'Sales', 'Seasonality', 'Trend', 'Noise']]
    
    # Transformation settings
    st.header("Feature Transformations")
    
    with st.expander("Adstock Transformation (Carryover Effect)", expanded=True):
        apply_adstock_transform = st.checkbox("Apply Adstock Transformation", True)
        
        if apply_adstock_transform:
            adstock_method = st.radio(
                "Adstock Method",
                ["Standard Geometric Decay", "Hill Adstock (Delayed Peak)"],
                index=0
            )
            
            adstock_rates = {}
            
            if adstock_method == "Standard Geometric Decay":
                col1, col2 = st.columns(2)
                
                with col1:
                    # Option for same rate for all channels
                    use_same_rate = st.checkbox("Use same decay rate for all channels", True)
                    
                    if use_same_rate:
                        global_decay_rate = st.slider(
                            "Global Decay Rate",
                            min_value=0.1,
                            max_value=0.9,
                            value=0.3,
                            help="Higher values mean faster decay of marketing effect"
                        )
                        
                        # Apply to all channels
                        for channel in channel_names:
                            adstock_rates[channel] = global_decay_rate
                
                if not use_same_rate:
                    st.markdown("##### Channel-Specific Decay Rates")
                    
                    # Use columns to make it more compact
                    cols = st.columns(min(3, len(channel_names)))
                    
                    for i, channel in enumerate(channel_names):
                        with cols[i % len(cols)]:
                            adstock_rates[channel] = st.slider(
                                f"{channel} Decay Rate",
                                min_value=0.1,
                                max_value=0.9,
                                value=0.3,
                                key=f"adstock_{channel}"
                            )
            else:  # Hill Adstock
                col1, col2 = st.columns(2)
                
                with col1:
                    # Global settings for Hill Adstock
                    global_decay_rate = st.slider(
                        "Decay Rate",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.3,
                        help="Rate at which marketing effect decays after peak"
                    )
                
                with col2:
                    peak_lag = st.slider(
                        "Peak Lag (periods)",
                        min_value=1,
                        max_value=8,
                        value=2,
                        help="Number of periods until marketing effect reaches peak"
                    )
                
                # Apply to all channels
                for channel in channel_names:
                    adstock_rates[channel] = {
                        'decay_rate': global_decay_rate,
                        'lag': peak_lag
                    }
                
                # Example visualization of Hill adstock effect
                if st.checkbox("Show example of Hill adstock effect"):
                    # Create a pulse input
                    x = np.zeros(20)
                    x[0] = 100  # Single pulse at t=0
                    
                    # Apply Hill adstock with different lags
                    lags = [1, 2, 4, 6]
                    fig = go.Figure()
                    
                    for lag in lags:
                        y = apply_hill_adstock(x, decay_rate=global_decay_rate, lag=lag)
                        fig.add_trace(go.Scatter(
                            x=list(range(len(x))),
                            y=y,
                            mode='lines',
                            name=f'Lag = {lag}'
                        ))
                    
                    fig.update_layout(
                        title="Hill Adstock Effect with Different Lags",
                        xaxis_title="Time Period",
                        yaxis_title="Effect",
                        legend_title="Peak Lag",
                        template="plotly_dark",
                        plot_bgcolor='rgba(25, 25, 44, 0.0)',
                        paper_bgcolor='rgba(25, 25, 44, 0.0)',
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Saturation Transformation (Diminishing Returns)", expanded=True):
        apply_saturation_transform = st.checkbox("Apply Saturation Transformation", True)
        
        if apply_saturation_transform:
            saturation_method = st.radio(
                "Saturation Method",
                ["Exponential Saturation", "S-Curve (Logistic)"],
                index=0
            )
            
            saturation_rates = {}
            
            if saturation_method == "Exponential Saturation":
                col1, col2 = st.columns(2)
                
                with col1:
                    # Option for same rate for all channels
                    use_same_rate = st.checkbox("Use same saturation rate for all channels", True)
                    
                    if use_same_rate:
                        global_sat_rate = st.slider(
                            "Global Saturation Rate",
                            min_value=0.01,
                            max_value=0.5,
                            value=0.1,
                            step=0.01,
                            help="Higher values mean quicker diminishing returns"
                        )
                        
                        # Apply to all channels
                        for channel in channel_names:
                            saturation_rates[channel] = global_sat_rate
                
                if not use_same_rate:
                    st.markdown("##### Channel-Specific Saturation Rates")
                    
                    # Use columns to make it more compact
                    cols = st.columns(min(3, len(channel_names)))
                    
                    for i, channel in enumerate(channel_names):
                        with cols[i % len(cols)]:
                            saturation_rates[channel] = st.slider(
                                f"{channel} Saturation",
                                min_value=0.01,
                                max_value=0.5,
                                value=0.1,
                                step=0.01,
                                key=f"saturation_{channel}"
                            )
            else:  # S-Curve
                col1, col2 = st.columns(2)
                
                with col1:
                    # Global settings for S-Curve
                    global_steepness = st.slider(
                        "Steepness",
                        min_value=0.01,
                        max_value=0.5,
                        value=0.1,
                        step=0.01,
                        help="Controls how steep the S-curve is"
                    )
                
                with col2:
                    inflection_point = st.slider(
                        "Inflection Point (%)",
                        min_value=10,
                        max_value=90,
                        value=50,
                        help="Where the curve changes from increasing to diminishing returns (as % of max spend)"
                    )
                
                # Apply to all channels
                for channel in channel_names:
                    # Calculate actual inflection point based on data range
                    channel_max = data[channel].max()
                    actual_inflection = channel_max * (inflection_point / 100)
                    
                    saturation_rates[channel] = {
                        'k': global_steepness,
                        'inflection': actual_inflection
                    }
                
                # Example visualization of S-curve effect
                if st.checkbox("Show example of S-curve effect"):
                    # Create a range of values
                    x = np.linspace(0, 1000, 100)
                    
                    # Apply S-curves with different parameters
                    steepness_values = [0.005, 0.01, 0.02, 0.05]
                    fig = go.Figure()
                    
                    for k in steepness_values:
                        y = apply_s_curve(x, k=k, inflection=500)
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            mode='lines',
                            name=f'Steepness = {k}'
                        ))
                    
                    fig.update_layout(
                        title="S-Curve Effect with Different Steepness Values",
                        xaxis_title="Input (Spend)",
                        yaxis_title="Output (Effect)",
                        legend_title="Steepness",
                        template="plotly_dark",
                        plot_bgcolor='rgba(25, 25, 44, 0.0)',
                        paper_bgcolor='rgba(25, 25, 44, 0.0)',
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Transform data based on settings
    if st.button("Apply Transformations", key="transform_button"):
        with st.spinner("Applying transformations..."):
            # Create transformed data
            transformed_data = data.copy()
            
            # Prepare transformation parameters
            adstock_params = {channel: {'decay_rate': rate} if isinstance(rate, float) else rate 
                             for channel, rate in adstock_rates.items()} if apply_adstock_transform else None
            
            saturation_params = {channel: {'k': rate} if isinstance(rate, float) else rate 
                                for channel, rate in saturation_rates.items()} if apply_saturation_transform else None
            
            # Apply transformations
            transformed_data = transform_marketing_data(
                transformed_data,
                channel_names,
                adstock_params=adstock_params,
                saturation_params=saturation_params
            )
            
            # Store in session state
            st.session_state.transformed_data = transformed_data
            
            # Get feature columns for modeling
            if apply_adstock_transform and apply_saturation_transform:
                # Use saturation columns (applied to adstock columns)
                feature_cols = [f"{channel}_sat" for channel in channel_names]
            elif apply_adstock_transform:
                # Use adstock columns
                feature_cols = [f"{channel}_adstock" for channel in channel_names]
            elif apply_saturation_transform:
                # Use saturation columns (applied to original columns)
                feature_cols = [f"{channel}_sat" for channel in channel_names]
            else:
                # Use original channels
                feature_cols = channel_names.copy()
            
            # Add seasonality and trend if they exist
            if 'Seasonality' in transformed_data.columns:
                feature_cols.append('Seasonality')
            if 'Trend' in transformed_data.columns:
                feature_cols.append('Trend')
            
            # Store feature columns in session state
            st.session_state.feature_cols = feature_cols
            
            st.success("Transformations applied successfully!")
    
    # Display transformations if they exist
    if hasattr(st.session_state, 'transformed_data') and st.session_state.transformed_data is not None:
        st.header("Transformed Data")
        
        # Select a channel to visualize
        selected_channel = st.selectbox(
            "Select a channel to visualize transformations:",
            channel_names
        )
        
        # Get the data
        transformed_data = st.session_state.transformed_data
        
        # Plot the original and transformed series
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=transformed_data['Date'],
            y=transformed_data[selected_channel],
            mode='lines',
            name='Original'
        ))
        
        # Adstock transformation
        if apply_adstock_transform and f"{selected_channel}_adstock" in transformed_data.columns:
            fig.add_trace(go.Scatter(
                x=transformed_data['Date'],
                y=transformed_data[f"{selected_channel}_adstock"],
                mode='lines',
                name='After Adstock'
            ))
        
        # Saturation transformation
        if apply_saturation_transform and f"{selected_channel}_sat" in transformed_data.columns:
            fig.add_trace(go.Scatter(
                x=transformed_data['Date'],
                y=transformed_data[f"{selected_channel}_sat"],
                mode='lines',
                name='After Saturation'
            ))
        
        fig.update_layout(
            title=f"Transformations for {selected_channel}",
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Transformation",
            template="plotly_dark",
            plot_bgcolor='rgba(25, 25, 44, 0.0)',
            paper_bgcolor='rgba(25, 25, 44, 0.0)',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model training
        st.header("Model Training")
        
        # Model selection
        st.subheader("Model Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_linear = st.checkbox("Linear Regression", True)
            train_ridge = st.checkbox("Ridge Regression", True)
        
        with col2:
            train_lasso = st.checkbox("Lasso Regression", False)
            train_elasticnet = st.checkbox("ElasticNet Regression", False)
        
        # Hyperparameters
        st.subheader("Hyperparameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if train_ridge:
                ridge_alpha = st.slider(
                    "Ridge Alpha",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.01
                )
            
            if train_lasso:
                lasso_alpha = st.slider(
                    "Lasso Alpha",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.01
                )
        
        with col2:
            if train_elasticnet:
                elasticnet_alpha = st.slider(
                    "ElasticNet Alpha",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.01
                )
                
                elasticnet_l1_ratio = st.slider(
                    "ElasticNet L1 Ratio",
                    min_value=0.01,
                    max_value=0.99,
                    value=0.5,
                    step=0.01
                )
        
        # Train models
        if st.button("Train Models", key="train_models_button"):
            with st.spinner("Training models..."):
                # Get features and target
                X = transformed_data[st.session_state.feature_cols]
                y = transformed_data['Sales']
                
                # Initialize models dictionary
                models = {}
                metrics = {}
                
                # Train linear regression
                if train_linear:
                    linear_model, linear_preds = train_linear_model(X, y)
                    models['Linear Regression'] = linear_model
                    metrics['Linear Regression'] = calculate_metrics(y, linear_preds)
                
                # Train ridge regression
                if train_ridge:
                    ridge_model, ridge_preds = train_ridge_model(X, y, alpha=ridge_alpha)
                    models['Ridge Regression'] = ridge_model
                    metrics['Ridge Regression'] = calculate_metrics(y, ridge_preds)
                
                # Train lasso regression
                if train_lasso:
                    lasso_model, lasso_preds = train_lasso_model(X, y, alpha=lasso_alpha)
                    models['Lasso Regression'] = lasso_model
                    metrics['Lasso Regression'] = calculate_metrics(y, lasso_preds)
                
                # Train elasticnet regression
                if train_elasticnet:
                    elasticnet_model, elasticnet_preds = train_elasticnet_model(
                        X, y, alpha=elasticnet_alpha, l1_ratio=elasticnet_l1_ratio
                    )
                    models['ElasticNet Regression'] = elasticnet_model
                    metrics['ElasticNet Regression'] = calculate_metrics(y, elasticnet_preds)
                
                # Store in session state
                st.session_state.models = models
                st.session_state.metrics = metrics
                
                st.success("Models trained successfully!")
    
    # Display model results if they exist
    if hasattr(st.session_state, 'models') and st.session_state.models:
        st.header("Model Results")
        
        # Display metrics
        st.subheader("Model Performance Metrics")
        
        metrics_df = pd.DataFrame(st.session_state.metrics).T
        st.dataframe(metrics_df.style.highlight_max(axis=0))
        
        # Select a model to visualize
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox(
            "Select a model to visualize:",
            model_names
        )
        
        model = st.session_state.models[selected_model]
        
        # Display model coefficients
        st.subheader("Model Coefficients")
        
        fig = plot_model_coefficients(model, st.session_state.feature_cols)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display actual vs predicted
        st.subheader("Actual vs Predicted")
        
        X = st.session_state.transformed_data[st.session_state.feature_cols]
        y = st.session_state.transformed_data['Sales']
        
        y_pred = model.predict(X)
        
        fig = plot_actual_vs_predicted(
            dates=st.session_state.transformed_data['Date'],
            y_true=y,
            y_pred=y_pred
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual analysis
        st.subheader("Residual Analysis")
        
        residuals = y - y_pred
        
        # Residuals over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=st.session_state.transformed_data['Date'],
            y=residuals,
            mode='lines',
            name='Residuals'
        ))
        
        fig.add_trace(go.Scatter(
            x=st.session_state.transformed_data['Date'],
            y=np.zeros_like(residuals),
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.5)', dash='dash'),
            name='Zero Line'
        ))
        
        fig.update_layout(
            title="Residuals Over Time",
            xaxis_title="Date",
            yaxis_title="Residual",
            template="plotly_dark",
            plot_bgcolor='rgba(25, 25, 44, 0.0)',
            paper_bgcolor='rgba(25, 25, 44, 0.0)',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual histogram
        fig = px.histogram(
            residuals,
            nbins=30,
            labels={'value': 'Residual'},
            title="Residual Distribution"
        )
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(25, 25, 44, 0.0)',
            paper_bgcolor='rgba(25, 25, 44, 0.0)',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Save model to session state for other pages
        st.session_state.selected_model = model
        st.session_state.selected_model_name = selected_model
        
        # Save all necessary data for the results page
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.y_pred = y_pred

if __name__ == "__main__":
    st.set_page_config(
        page_title="Model Training",
        page_icon="🧠",
        layout="wide"
    )
    render_model_training()