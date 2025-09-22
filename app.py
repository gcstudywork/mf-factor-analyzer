import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
import plotly.graph_objects as go
from mfapi import MFAPI
from indiafactorlibrary import IndiaFactorLibrary

# Page config
st.set_page_config(
    page_title="MF Return Decomposition Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize
mf = MFAPI()

# Title
st.title("Return Decomposition Analysis")

# Sidebar
with st.sidebar:
    st.header("Fund Selection")
    
    # Get schemes
    with st.spinner("Loading schemes..."):
        scheme_codes = mf.get_scheme_codes()
    
    if scheme_codes:
        scheme_names = {v: k for k, v in scheme_codes.items()}
        
        # Search
        search_term = st.text_input("Search fund name", "")
        
        if search_term:
            filtered_names = [name for name in scheme_names.keys() 
                            if search_term.lower() in name.lower()]
        else:
            filtered_names = list(scheme_names.keys())[:100]
        
        # Select scheme
        if filtered_names:
            selected_name = st.selectbox(
                "Select Fund",
                options=filtered_names,
                key="fund_select"
            )
            
            if selected_name:
                scheme_code = scheme_names[selected_name]
                
                # Show basic details
                details = mf.get_scheme_details(scheme_code)
                if details:
                    st.divider()
                    st.subheader("Fund Details")
                    st.write(f"**Fund House:** {details['fund_house']}")
                    st.write(f"**Category:** {details['scheme_category']}")
                    st.write(f"**Latest NAV:** ₹{details['nav']}")

# Main area
if selected_name:
    try:
        # Initialize India Factor Library
        ifl = IndiaFactorLibrary()
        
        # Read the Fama-French 6 factors dataset (monthly returns)
        ff6 = ifl.read('ff6')[0]
        
        # Get fund data
        fund_data = mf.get_scheme_historical_nav(scheme_code)
        
        if not fund_data:
            st.write('No data available for this scheme.')
            st.stop()
            
        # Process fund data
        fund_df = pd.DataFrame(fund_data)
        
        # Handle date format - fix for DD-MM-YYYY format
        try:
            fund_df['date'] = pd.to_datetime(fund_df['date'], dayfirst=True)
        except:
            # Fallback to explicit format
            fund_df['date'] = pd.to_datetime(fund_df['date'], format='%d-%m-%Y')
        
        fund_df['nav'] = pd.to_numeric(fund_df['nav'], errors='coerce')
        fund_df = fund_df.dropna()
        
        if fund_df.empty:
            st.error("No valid data after processing. Check data format.")
            st.stop()
        
        # Get min and max dates for period selection
        min_date = fund_df['date'].min()
        max_date = fund_df['date'].max()
        
        st.sidebar.subheader("📅 Analysis Period")
        
        # Create date inputs with min/max constraints
        start_date = st.sidebar.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # Convert to datetime for comparison
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Validate date range
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()
        
        # Filter data based on selected period
        fund_df = fund_df[(fund_df['date'] >= start_date) & (fund_df['date'] <= end_date)]
        
        if fund_df.empty:
            st.error("No data available for the selected period.")
            st.stop()
        
        # Set date as index and resample to monthly
        fund_df = fund_df.set_index('date')
        fund_monthly = fund_df.resample('ME').last()
        
        # Calculate monthly returns
        fund_monthly['returns'] = fund_monthly['nav'].pct_change() * 100
        fund_monthly = fund_monthly.dropna()
        
        # Check if we have enough data points
        n_months = len(fund_monthly)
        if n_months < 36:
            st.warning(f"⚠️ Limited data: Only {n_months} months available (minimum 36 recommended for reliable analysis)")
        
        # Align with factor data
        ff6.index = pd.to_datetime(ff6.index)
        
        # Filter factor data to match selected period
        ff6 = ff6[(ff6.index >= start_date) & (ff6.index <= end_date)]
        
        merged_data = fund_monthly.merge(ff6, left_index=True, right_index=True, how='inner')
        
        if merged_data.empty:
            st.error("No overlapping data between fund and factors for the selected period.")
            st.stop()
        
        # Check observations after merging
        n_observations = len(merged_data)
        if n_observations < 36:
            st.warning(f"⚠️ Limited observations: Only {n_observations} overlapping data points (minimum 36 recommended)")
        
        # Calculate excess returns and prepare for regression
        merged_data['excess_return'] = merged_data['returns'] - merged_data['RF']
        factors = ['MF', 'SMB5', 'HML', 'RMW', 'CMA', 'WML']
        regression_data = merged_data[['excess_return'] + factors].dropna()
        
        # Final observation check
        final_obs = len(regression_data)
        st.write(f"**Period Analyzed**: {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
        st.write(f"**Number of Observations**: {final_obs} months")
        
        if final_obs < 24:
            st.error("Insufficient data points for meaningful analysis. Please select a longer period.")
            st.stop()
        
        # Prepare regression data (scale percentages to decimals)
        X = regression_data[factors]
        X = sm.add_constant(X)
        X_scaled = X / 100  # Scale factors to decimals
        y_scaled = regression_data['excess_return'] / 100  # Scale returns to decimals
        
        # Perform regression using statsmodels
        model = sm.OLS(y_scaled, X_scaled).fit()
        
        # Display basic results
        st.subheader("📊 Model Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R-squared", f"{model.rsquared:.3f}")
        with col2:
            st.metric("Observations", final_obs)
        with col3:
            st.metric("Alpha", f"{model.params['const']:.4f}")
        with col4:
            p_value = model.pvalues['const']
            sig_status = "✅" if p_value < 0.05 else "⚠️" if p_value < 0.1 else "❌"
            st.metric("Alpha Significance", sig_status, f"p={p_value:.3f}")

        # Factor exposures chart
        st.subheader("📈 Factor Exposures")
        
        # Get coefficients for factors only
        factor_coeffs = model.params[1:7]
        factor_pvalues = model.pvalues[1:7]
        
        # Create bar chart using plotly graph_objects (to match your original style)
        fig = go.Figure()
        
        # Add bars with color based on significance
        colors = ['blue' if pval < 0.05 else 'red' for pval in factor_pvalues]
        
        fig.add_trace(go.Bar(
            x=factor_coeffs.index,
            y=factor_coeffs.values,
            marker_color=colors,
            text=[f'{coef:.4f}' for coef in factor_coeffs.values],
            textposition='auto',
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(
            title="Factor Coefficients",
            xaxis_title="Factor",
            yaxis_title="Exposure",
            showlegend=False
        )
        
        st.plotly_chart(fig)

        # Interpretation
        st.subheader("💡 Key Insights")
        
        # Market exposure
        market_beta = model.params['MF']
        if market_beta > 1.0:
            st.write(f"• **Market Sensitive**: Moves {market_beta:.2f} times more than the market")
        elif market_beta > 0.8:
            st.write(f"• **Market Neutral**: Moves {market_beta:.2f} times with the market")
        else:
            st.write(f"• **Market Defensive**: Moves {market_beta:.2f} times less than the market")
        
        # Check if any factors are significant
        significant_factors = [factor for factor in factors if model.pvalues[factor] < 0.05]
        if significant_factors:
            st.write(f"• **Significant Factors**: {', '.join(significant_factors)}")
        else:
            st.write("• No factors show statistically significant exposure")

        # Performance chart
        st.subheader("📅 Performance Trend")
        
        # Calculate predicted returns (scale back to percentages)
        regression_data['predicted'] = model.predict(X_scaled) * 100
        
        # Create line chart
        performance_df = regression_data[['excess_return', 'predicted']].copy()
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=performance_df.index,
            y=performance_df['excess_return'],
            mode='lines',
            name='Actual Excess Return',
            line=dict(color='blue', width=2)
        ))
        fig_perf.add_trace(go.Scatter(
            x=performance_df.index,
            y=performance_df['predicted'],
            mode='lines',
            name='Predicted',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig_perf.update_layout(
            title="Actual vs Predicted Excess Returns",
            xaxis_title="Date",
            yaxis_title="Excess Return (%)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig_perf)

        # Data quality info
        with st.expander("📋 Data Quality Information"):
            st.write(f"**Fund Data Range**: {min_date.strftime('%d %b %Y')} to {max_date.strftime('%d %b %Y')}")
            st.write(f"**Selected Period**: {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
            st.write(f"**Available Months**: {n_months}")
            st.write(f"**Overlapping Observations**: {n_observations}")
            st.write(f"**Final Observations**: {final_obs}")
            
            if final_obs < 36:
                st.warning("For more reliable results, consider selecting a longer time period (3+ years)")
        
    except Exception as e:
        st.error(f"Error in factor analysis: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
else:
    st.info("👈 Select a fund from the sidebar to begin analysis")