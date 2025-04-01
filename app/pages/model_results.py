import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mmm.evaluation import decompose_contributions, calculate_roi, calculate_elasticity
from app.components.visualizations import plot_contributions, plot_roi_bar_chart

def render_model_results():
    """
    Render the model results page.
    """
    st.title("Model Results & Analysis")
    
    # Check if we have trained models
    if not hasattr(st.session_state, 'models') or not st.session_state.models:
        st.warning("No trained models available. Please train models first.")
        return
    
    # Check if we have all required data
    required_attributes = ['data', 'transformed_data', 'feature_cols', 'selected_model']
    missing_attributes = [attr for attr in required_attributes if not hasattr(st.session_state, attr)]
    
    if missing_attributes:
        st.warning(f"Missing required data: {', '.join(missing_attributes)}. Please complete the model training process.")
        return
    
    # Get data and model
    data = st.session_state.data
    transformed_data = st.session_state.transformed_data
    feature_cols = st.session_state.feature_cols
    model = st.session_state.selected_model
    model_name = st.session_state.selected_model_name
    
    # Get channel names
    channel_names = [col for col in data.columns if col not in ['Date', 'Sales', 'Seasonality', 'Trend', 'Noise']]
    
    # Model summary
    st.header("Model Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Selected Model", model_name)
        
        if hasattr(st.session_state, 'metrics'):
            metrics = st.session_state.metrics[model_name]
            st.metric("R² Score", f"{metrics['R2']:.4f}")
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
    
    with col2:
        st.metric("Marketing Channels", len(channel_names))
        
        if hasattr(model, 'coef_'):
            # Count significant channels (non-zero coefficients)
            significant_channels = sum(abs(coef) > 1e-5 for coef in model.coef_[:len(channel_names)])
            st.metric("Significant Channels", significant_channels)
            
            # Count negative coefficients
            negative_channels = sum(coef < 0 for coef in model.coef_[:len(channel_names)])
            st.metric("Channels with Negative Effect", negative_channels)
    
    # Decomposition analysis
    st.header("Decomposition Analysis")
    
    # Get the decomposition of sales
    contributions = decompose_contributions(model, transformed_data[feature_cols], original_channels=channel_names)
    
    # Add date and actual sales for plotting
    contributions['Date'] = transformed_data['Date']
    contributions['Actual_Sales'] = transformed_data['Sales']
    
    # Store in session state
    st.session_state.contributions = contributions
    
    # Plot stacked area chart of contributions
    st.subheader("Sales Decomposition")
    
    fig = plot_contributions(contributions, channel_names)
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate channel contribution metrics
    st.subheader("Channel Contribution Summary")
    
    # Calculate total contribution per channel
    channel_totals = {}
    for col in channel_names:
        if col in contributions.columns:
            channel_totals[col] = contributions[col].sum()
    
    # Calculate total contribution and percentage
    total_contrib = sum(filter(lambda x: x > 0, channel_totals.values()))  # Sum only positive contributions
    
    # Convert to DataFrame for visualization
    contribution_df = pd.DataFrame({
        'Channel': list(channel_totals.keys()),
        'Total Contribution': list(channel_totals.values())
    })
    
    contribution_df['Contribution %'] = (contribution_df['Total Contribution'] / total_contrib * 100).round(2)
    contribution_df = contribution_df.sort_values('Total Contribution', ascending=False)
    
    # Display as a table with conditional formatting
    st.dataframe(
        contribution_df.style.background_gradient(subset=['Contribution %'], cmap='viridis'),
        use_container_width=True
    )
    
    # Pie chart of channel contributions
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Filter to only positive contributions for the pie chart
        positive_contrib_df = contribution_df[contribution_df['Total Contribution'] > 0].copy()
        
        fig = px.pie(
            positive_contrib_df,
            values='Total Contribution',
            names='Channel',
            title='Channel Contribution Share',
            hole=0.4
        )
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(25, 25, 44, 0.0)',
            paper_bgcolor='rgba(25, 25, 44, 0.0)',
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Calculate base vs marketing contribution
        if 'Intercept' in contributions.columns:
            base_contribution = contributions['Intercept'].mean()
        else:
            base_contribution = 0
        
        marketing_contribution = sum(channel_totals.values())
        
        # Calculate percentages
        total_sales = base_contribution + marketing_contribution
        base_pct = (base_contribution / total_sales * 100).round(2)
        marketing_pct = (marketing_contribution / total_sales * 100).round(2)
        
        # Create a pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Base Sales', 'Marketing Driven'],
            values=[base_contribution, marketing_contribution],
            hole=0.4,
            marker=dict(
                colors=['rgba(100, 100, 255, 0.7)', 'rgba(200, 100, 255, 0.7)'],
                line=dict(color='rgba(0, 0, 0, 0)', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=12, color='white'),
            insidetextorientation='radial'
        )])
        
        fig.update_layout(
            title='Base vs Marketing Driven Sales',
            template="plotly_dark",
            plot_bgcolor='rgba(25, 25, 44, 0.0)',
            paper_bgcolor='rgba(25, 25, 44, 0.0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ROI analysis
    st.header("ROI Analysis")
    
    # Calculate ROI
    roi_df = calculate_roi(model, transformed_data[feature_cols], data, channel_names)
    
    # Sort by ROI
    roi_df = roi_df.sort_values('ROI', ascending=False)
    
    # Display as a table with conditional formatting
    st.dataframe(
        roi_df.style.background_gradient(subset=['ROI'], cmap='viridis'),
        use_container_width=True
    )
    
    # ROI bar chart
    fig = plot_roi_bar_chart(roi_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Spend vs. contribution scatter plot
    st.subheader("Spend vs. Contribution Analysis")
    
    fig = px.scatter(
        roi_df,
        x='Total Spend',
        y='Total Contribution',
        size='ROI',
        color='ROI',
        hover_name='Channel',
        color_continuous_scale='viridis',
        size_max=50,
        labels={
            'Total Spend': 'Total Marketing Spend',
            'Total Contribution': 'Total Sales Contribution',
            'ROI': 'Return on Investment'
        }
    )
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(25, 25, 44, 0.0)',
        paper_bgcolor='rgba(25, 25, 44, 0.0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Elasticity analysis
    st.header("Elasticity Analysis")
    
    # Calculate elasticity
    elasticity_df = calculate_elasticity(model, transformed_data[feature_cols], channel_names)
    
    # Sort by elasticity
    elasticity_df = elasticity_df.sort_values('Elasticity', ascending=False)
    
    # Display as a table
    st.dataframe(elasticity_df)
    
    # Elasticity bar chart
    fig = px.bar(
        elasticity_df,
        x='Channel',
        y='Elasticity',
        labels={'Channel': 'Marketing Channel', 'Elasticity': 'Sales Elasticity'},
        color='Elasticity',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        title='Sales Elasticity by Channel',
        template="plotly_dark",
        plot_bgcolor='rgba(25, 25, 44, 0.0)',
        paper_bgcolor='rgba(25, 25, 44, 0.0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # What-if analysis
    st.header("What-If Analysis")
    
    st.markdown("""
    Use the sliders below to adjust marketing spend for each channel and see the predicted impact on sales.
    """)
    
    # Base values for what-if analysis
    base_values = {}
    for channel in channel_names:
        base_values[channel] = data[channel].mean()
    
    # Create sliders for each channel
    what_if_values = {}
    
    for i, channel in enumerate(channel_names):
        default_value = int(base_values[channel])
        min_value = int(max(0, default_value * 0.5))
        max_value = int(default_value * 1.5)
        
        what_if_values[channel] = st.slider(
            f"{channel} Spend",
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=int((max_value - min_value) / 100) or 1
        )
    
    # Create a DataFrame with the what-if values
    what_if_data = transformed_data.copy()
    
    for channel in channel_names:
        what_if_data[channel] = what_if_values[channel]
    
    # Apply transformations if needed
    # For simplicity, we'll just use the existing transformations in the transformed_data
    
    # Make predictions
    X_what_if = what_if_data[feature_cols]
    y_what_if = model.predict(X_what_if)
    
    # Calculate baseline prediction (current average)
    y_baseline = data['Sales'].mean()
    
    # Display results
    st.subheader("What-If Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Average Sales",
            f"{y_baseline:.2f}"
        )
    
    with col2:
        st.metric(
            "Predicted Sales",
            f"{y_what_if.mean():.2f}"
        )
    
    with col3:
        # Calculate percentage change
        pct_change = ((y_what_if.mean() / y_baseline) - 1) * 100
        
        st.metric(
            "Change",
            f"{pct_change:.2f}%",
            delta=f"{pct_change:.2f}%"
        )
    
    # Export results
    st.header("Export Results")
    
    # Create a DataFrame with all results
    results = {
        'Model': model_name,
        'R2': st.session_state.metrics[model_name]['R2'],
        'RMSE': st.session_state.metrics[model_name]['RMSE'],
        'MAE': st.session_state.metrics[model_name]['MAE'],
        'Base Sales': base_contribution if 'Intercept' in contributions.columns else 0,
        'Base Sales %': base_pct if 'Intercept' in contributions.columns else 0,
        'Marketing Driven Sales': marketing_contribution,
        'Marketing Driven %': marketing_pct
    }
    
    # Add channel contributions
    for channel in channel_names:
        results[f'{channel} Contribution'] = channel_totals.get(channel, 0)
        results[f'{channel} Contribution %'] = contribution_df.loc[contribution_df['Channel'] == channel, 'Contribution %'].values[0] if channel in contribution_df['Channel'].values else 0
        results[f'{channel} ROI'] = roi_df.loc[roi_df['Channel'] == channel, 'ROI'].values[0] if channel in roi_df['Channel'].values else 0
        results[f'{channel} Elasticity'] = elasticity_df.loc[elasticity_df['Channel'] == channel, 'Elasticity'].values[0] if channel in elasticity_df['Channel'].values else 0
    
    # Convert to DataFrame
    results_df = pd.DataFrame([results])
    
    # Download button
    st.download_button(
        "Download Results as CSV",
        results_df.to_csv(index=False),
        "mmm_results.csv",
        "text/csv",
        key='download-results-csv'
    )
    
    # Also allow downloading the contributions
    st.download_button(
        "Download Contributions as CSV",
        contributions.to_csv(index=False),
        "mmm_contributions.csv",
        "text/csv",
        key='download-contributions-csv'
    )

if __name__ == "__main__":
    st.set_page_config(
        page_title="Model Results",
        page_icon="📈",
        layout="wide"
    )
    render_model_results()