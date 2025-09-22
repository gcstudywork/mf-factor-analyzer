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
        
        # Read the Fama-French 6 factors dataset (monthly returns in decimals)
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
        
        # Calculate monthly returns (convert to percentage points for consistency)
        fund_monthly['returns'] = fund_monthly['nav'].pct_change() * 100
        
        # Remove any infinite values and drop NA
        fund_monthly = fund_monthly.replace([np.inf, -np.inf], np.nan)
        fund_monthly = fund_monthly.dropna()
        
        # Check if we have enough data points
        n_months = len(fund_monthly)
        if n_months < 36:
            st.warning(f"⚠️ Limited data: Only {n_months} months available (minimum 36 recommended for reliable analysis)")
        
        # Align with factor data
        ff6.index = pd.to_datetime(ff6.index)
        
        # Filter factor data to match selected period
        ff6 = ff6[(ff6.index >= start_date) & (ff6.index <= end_date)]
        
        # Convert factor data from decimals to percentage points (0.5 → 0.5%)
        # Since factors are already in percentage terms (0.5 means 0.5%), we keep as is
        # But ensure consistency with fund returns which are also in percentage points
        
        merged_data = fund_monthly.merge(ff6, left_index=True, right_index=True, how='inner')
        
        if merged_data.empty:
            st.error("No overlapping data between fund and factors for the selected period.")
            st.stop()
        
        # Check observations after merging
        n_observations = len(merged_data)
        if n_observations < 36:
            st.warning(f"⚠️ Limited observations: Only {n_observations} overlapping data points (minimum 36 recommended)")
        
        # Calculate excess returns (both already in percentage points)
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
        
        # Prepare regression data - NO SCALING needed since both are already in percentage points
        X = regression_data[factors]
        X = sm.add_constant(X)
        y = regression_data['excess_return']
        
        # Perform regression using statsmodels (data already in correct units)
        model = sm.OLS(y, X).fit()
        
        # Display basic results
        st.subheader("📊 Model Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R-squared", f"{model.rsquared:.3f}")
        with col2:
            st.metric("Observations", final_obs)
        with col3:
            st.metric("Alpha", f"{model.params['const']:.3f}")
        with col4:
            p_value = model.pvalues['const']
            sig_status = "✅" if p_value < 0.05 else "⚠️" if p_value < 0.1 else "❌"
            st.metric("Alpha Significance", sig_status, f"p={p_value:.3f}")

        # Factor exposures chart (matching your image style)
        st.subheader("Factor Exposures")
        
        # Get coefficients and p-values
        coefficients = model.params
        pvalues = model.pvalues
        
        # Create significance markers
        sig_markers = []
        for pval in pvalues:
            if pval < 0.01:
                sig_markers.append('***')
            elif pval < 0.05:
                sig_markers.append('**')
            elif pval < 0.1:
                sig_markers.append('*')
            else:
                sig_markers.append('')
        
        # Factor names matching your chart
        factor_names = ['const<br>(Intercept)', 'MF<br>(Market)', 'SMB5<br>(Size)',
                       'HML<br>(Value)', 'RMW<br>(Profitability)',
                       'CMA<br>(Investment)', 'WML<br>(Momentum)']
        
        # Colors based on significance (blue for significant, gray for not)
        colors = ['lightblue' if p < 0.1 else 'lightgray' for p in pvalues]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=factor_names,
            y=coefficients.values,
            marker_color=colors,
            text=[f"{coef:.2f} {sig}" for coef, sig in zip(coefficients.values, sig_markers)],
            textposition='outside',
            textfont=dict(size=12, family="Arial, sans-serif"),
        ))
        
        fig.update_layout(
            title=dict(
                text="Factor Exposures",
                font=dict(size=20, family="Arial, sans-serif")
            ),
            xaxis=dict(
                title="Factor",
                title_font=dict(size=16, family="Arial, sans-serif"),
                tickfont=dict(size=12, family="Arial, sans-serif")
            ),
            yaxis=dict(
                title="Coefficient",
                title_font=dict(size=16, family="Arial, sans-serif"),
                tickfont=dict(size=12, family="Arial, sans-serif")
            ),
            showlegend=False,
            height=500,
            plot_bgcolor='white'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add significance note
        fig.add_annotation(
            text="<i>Significance levels: * p<0.1, ** p<0.05, *** p<0.01</i>",
            xref="paper", yref="paper",
            x=0, y=-0.15,
            showarrow=False,
            font=dict(size=10, family="Arial, sans-serif", color="gray")
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation (matching your image exactly)
        st.subheader("Interpretation")
        
        interpretations = []
        
        # R-squared interpretation
        if model.rsquared < 0.3:
            interpretations.append(f"The model has limited explanatory power (R² = {model.rsquared:.2f}).")
        elif model.rsquared < 0.7:
            interpretations.append(f"The model has moderate explanatory power (R² = {model.rsquared:.2f}).")
        else:
            interpretations.append(f"The model has strong explanatory power (R² = {model.rsquared:.2f}).")
        
        # Alpha interpretation
        alpha = model.params['const']
        alpha_pval = model.pvalues['const']
        if alpha_pval < 0.05:
            if alpha > 0:
                interpretations.append(f"The fund shows statistically significant alpha of {alpha:.3f}, suggesting skill or exposure not captured by the known factors.")
            else:
                interpretations.append(f"The fund shows statistically significant negative alpha of {alpha:.3f}, suggesting underperformance relative to factor exposures.")
        else:
            interpretations.append(f"The fund's alpha ({alpha:.3f}) is not statistically significant, suggesting performance is explained by factor exposures.")
        
        # Market factor interpretation (corrected logic)
        mf_coef = model.params['MF']
        if mf_coef > 1.2:
            interpretations.append(f"**Market:** The fund has high sensitivity to how the broader market moves.")
        elif mf_coef > 0.8:
            interpretations.append(f"**Market:** The fund moves in line with the broader market.")
        else:
            interpretations.append(f"**Market:** The fund has low sensitivity to how the broader market moves.")
        
        # Size factor (SMB5)
        if model.pvalues['SMB5'] < 0.1:
            if model.params['SMB5'] > 0:
                interpretations.append(f"**Size Factor (SMB):** The fund favors small-cap stocks—more exposure to firms with smaller market capitalization.")
            else:
                interpretations.append(f"**Size Factor (SMB):** The fund favors large-cap stocks—more exposure to firms with larger market capitalization.")
        
        # Value factor (HML)
        if model.pvalues['HML'] < 0.1:
            if model.params['HML'] > 0:
                interpretations.append(f"**Value Factor (HML):** The fund tilts toward value stocks—typically those with high book-to-market ratios.")
            else:
                interpretations.append(f"**Value Factor (HML):** The fund tilts toward growth stocks—typically those with low book-to-market ratios and high price multiples.")
        
        # Profitability factor (RMW)
        if model.pvalues['RMW'] < 0.1:
            if model.params['RMW'] > 0:
                interpretations.append(f"**Profitability Factor (RMW):** The fund is exposed to highly profitable firms.")
            else:
                interpretations.append(f"**Profitability Factor (RMW):** The fund is exposed to low profitability or unprofitable firms.")
        
        # Investment factor (CMA)
        if model.pvalues['CMA'] < 0.1:
            if model.params['CMA'] > 0:
                interpretations.append(f"**Investment Factor (CMA):** The fund tilts toward conservatively investing firms—those with low investment rates.")
            else:
                interpretations.append(f"**Investment Factor (CMA):** The fund tilts toward aggressively investing firms—those that reinvest heavily in expansion.")
        
        # Momentum factor (WML)
        if model.pvalues['WML'] < 0.1:
            if model.params['WML'] > 0:
                interpretations.append(f"**Momentum Factor (WML):** The fund has a momentum tilt, favoring stocks that have performed well recently.")
            else:
                interpretations.append(f"**Momentum Factor (WML):** The fund has a contrarian tilt, favoring stocks that have performed poorly recently.")
        
        for interp in interpretations:
            st.markdown(f"• {interp}")

        # Performance chart
        st.subheader("Actual vs Predicted Excess Returns")
        
        # Calculate predicted returns
        regression_data['predicted'] = model.predict(X)
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=regression_data.index,
            y=regression_data['excess_return'],
            mode='lines',
            name='Actual Excess Return',
            line=dict(color='blue', width=2)
        ))
        fig_perf.add_trace(go.Scatter(
            x=regression_data.index,
            y=regression_data['predicted'],
            mode='lines',
            name='Predicted',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig_perf.update_layout(
            title="Actual vs Predicted Excess Returns",
            xaxis_title="Date",
            yaxis_title="Excess Return (%)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=400
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)

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