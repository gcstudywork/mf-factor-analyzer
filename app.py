import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from mfapi import MFAPI
from indiafactorlibrary import IndiaFactorLibrary

# Page config
st.set_page_config(
    page_title="Return Decomposition Analysis",
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
        selected_name = None
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
        
        # Read the Fama-French 6 factors dataset
        ff6 = ifl.read('ff6')[0]
        
        # Get fund data
        fund_data = mf.get_scheme_historical_nav(scheme_code)
        
        if not fund_data:
            st.write('No data available for this scheme.')
            st.stop()
            
        # Process fund data
        fund_df = pd.DataFrame(fund_data)
        
        # Handle date format
        try:
            fund_df['date'] = pd.to_datetime(fund_df['date'], dayfirst=True)
        except:
            fund_df['date'] = pd.to_datetime(fund_df['date'], format='%d-%m-%Y')
        
        fund_df['nav'] = pd.to_numeric(fund_df['nav'], errors='coerce')
        fund_df = fund_df.dropna()
        
        if fund_df.empty:
            st.error("No valid data after processing.")
            st.stop()
        
        # Get min and max dates for period selection
        min_date = fund_df['date'].min()
        max_date = fund_df['date'].max()
        
        st.sidebar.subheader("📅 Analysis Period")
        
        # Create date inputs
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
        fund_monthly = fund_df.resample('M').last()  # Changed from 'ME' to 'M'
        
        # Calculate monthly returns (in percentage points)
        fund_monthly['returns'] = fund_monthly['nav'].pct_change() * 100
        
        # Remove any infinite values and drop NA
        fund_monthly = fund_monthly.replace([np.inf, -np.inf], np.nan)
        fund_monthly = fund_monthly.dropna()
        
        # Align with factor data
        ff6.index = pd.to_datetime(ff6.index)
        ff6 = ff6[(ff6.index >= start_date) & (ff6.index <= end_date)]
        
        # Merge data (both already in percentage points)
        merged_data = fund_monthly.merge(ff6, left_index=True, right_index=True, how='inner')
        
        if merged_data.empty:
            st.error("No overlapping data between fund and factors.")
            st.stop()
        
        # Calculate excess returns (matching notebook logic)
        merged_data['excess_return'] = merged_data['returns'] - merged_data['RF']
        
        # Prepare data for regression (matching notebook scaling)
        factors = ['MF', 'SMB5', 'HML', 'RMW', 'CMA', 'WML']
        regression_data = merged_data[['excess_return'] + factors].dropna()
        
        # Convert to decimals by dividing by 100 (CRITICAL FIX - matches notebook)
        regression_data_scaled = regression_data / 100
        
        # Final observation check
        final_obs = len(regression_data_scaled)
        
        if final_obs < 24:
            st.error("Insufficient data points for meaningful analysis. Please select a longer period.")
            st.stop()
        
        # Perform regression using formula API (matching notebook approach)
        formula = 'excess_return ~ MF + SMB5 + HML + RMW + CMA + WML'
        model = smf.ols(formula=formula, data=regression_data_scaled).fit()
        
        # Summary Statistics Table
        st.subheader("📊 Summary Statistics")
        
        summary_data = {
            "Factor Model": "Fama-French 5 Factor Model + Momentum",
            "Analysis Period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "Number of Observations": final_obs,
            "R-squared": f"{model.rsquared:.3f}",
            "Adjusted R-squared": f"{model.rsquared_adj:.3f}",
            "F-statistic": f"{model.fvalue:.1f}",
            "Prob (F-statistic)": f"{model.f_pvalue:.4f}"
        }
        
        summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
        st.table(summary_df.set_index('Metric'))

        # Factor Exposures Chart
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
        
        # Factor names
        factor_names = ['Intercept', 'MF (Market)', 'SMB5 (Size)',
                       'HML (Value)', 'RMW (Profitability)',
                       'CMA (Investment)', 'WML (Momentum)']
        
        # Colors based on significance
        colors = []
        for i, p in enumerate(pvalues):
            if i == 0:  # Intercept
                colors.append('#86B6F6' if p < 0.1 else '#D3D3D3')
            elif i == 1:  # Market
                colors.append('#FFA500' if p < 0.1 else '#D3D3D3')
            elif i == 2:  # Size
                colors.append('#90EE90' if p < 0.1 else '#D3D3D3')
            elif i == 3:  # Value
                colors.append('#FFE4B5' if p < 0.1 else '#D3D3D3')
            elif i == 4:  # Profitability
                colors.append('#DDA0DD' if p < 0.1 else '#D3D3D3')
            elif i == 5:  # Investment
                colors.append('#F0E68C' if p < 0.1 else '#D3D3D3')
            else:  # Momentum
                colors.append('#FFB6C1' if p < 0.1 else '#D3D3D3')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=factor_names,
            y=coefficients.values,
            marker_color=colors,
            text=[f"{coef:.4f} {sig}" for coef, sig in zip(coefficients.values, sig_markers)],
            textposition='outside',
            textfont=dict(size=12, family="Arial, sans-serif", color='blue'),
            error_y=dict(
                type='data',
                array=[model.bse.iloc[i] for i in range(len(coefficients))],
                visible=True
            )
        ))
        
        fig.update_layout(
            title=dict(
                text="Factor Exposures (Coefficients in Decimal Form)",
                font=dict(size=20, family="Arial, sans-serif")
            ),
            xaxis=dict(
                title="Factor",
                title_font=dict(size=14, family="Arial, sans-serif"),
                tickfont=dict(size=12, family="Arial, sans-serif")
            ),
            yaxis=dict(
                title="Coefficient",
                title_font=dict(size=14, family="Arial, sans-serif"),
                tickfont=dict(size=12, family="Arial, sans-serif")
            ),
            showlegend=False,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
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

        # Multicollinearity Check (VIF)
        st.subheader("🔍 Multicollinearity Check (VIF)")
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Calculate VIF for each factor
        X_factors = regression_data_scaled[factors]
        vifs = pd.DataFrame({
            'Factor': factors,
            'VIF': [variance_inflation_factor(X_factors.values, i) for i in range(len(factors))]
        })
        
        st.write("Variance Inflation Factors (VIF > 5-10 suggests multicollinearity):")
        st.dataframe(vifs.round(2))

        # Interpretation Section
        st.subheader("Interpretation")
        
        interpretations = []
        
        # R-squared interpretation
        if model.rsquared < 0.3:
            interpretations.append(f"The model has limited explanatory power (R² = {model.rsquared:.3f}).")
        elif model.rsquared < 0.7:
            interpretations.append(f"The model has moderate explanatory power (R² = {model.rsquared:.3f}).")
        else:
            interpretations.append(f"The model has strong explanatory power (R² = {model.rsquared:.3f}).")
        
        # Alpha interpretation
        alpha = model.params['Intercept']
        alpha_pval = model.pvalues['Intercept']
        if alpha_pval < 0.05:
            if alpha > 0:
                interpretations.append(f"**Alpha:** The fund shows statistically significant positive alpha of {alpha:.4f}, suggesting skill or exposure not captured by the known factors.")
            else:
                interpretations.append(f"**Alpha:** The fund shows statistically significant negative alpha of {alpha:.4f}, suggesting underperformance relative to factor exposures.")
        else:
            interpretations.append(f"**Alpha:** The fund's alpha ({alpha:.4f}) is not statistically significant.")
        
        # Market factor
        mf_coef = model.params['MF']
        mf_pval = model.pvalues['MF']
        if mf_pval < 0.1:
            if mf_coef > 1.1:
                interpretations.append(f"**Market Factor (MF):** The fund has high sensitivity to market movements.")
            elif mf_coef > 0.9:
                interpretations.append(f"**Market Factor (MF):** The fund moves in line with the broader market.")
            elif mf_coef > 0:
                interpretations.append(f"**Market Factor (MF):** The fund has low sensitivity to market movements.")
        
        # Size factor (SMB5)
        smb_coef = model.params['SMB5']
        smb_pval = model.pvalues['SMB5']
        if smb_pval < 0.1:
            if smb_coef > 0:
                interpretations.append(f"**Size Factor (SMB5):** The fund favors small-cap stocks.")
            else:
                interpretations.append(f"**Size Factor (SMB5):** The fund favors large-cap stocks.")
        
        # Value factor (HML)
        hml_coef = model.params['HML']
        hml_pval = model.pvalues['HML']
        if hml_pval < 0.1:
            if hml_coef > 0:
                interpretations.append(f"**Value Factor (HML):** The fund tilts toward value stocks.")
            else:
                interpretations.append(f"**Value Factor (HML):** The fund tilts toward growth stocks.")
        else:
            interpretations.append(f"**Value Factor (HML):** The fund shows no significant tilt toward value or growth stocks.")
        
        # Profitability factor (RMW)
        rmw_coef = model.params['RMW']
        rmw_pval = model.pvalues['RMW']
        if rmw_pval < 0.1:
            if rmw_coef > 0:
                interpretations.append(f"**Profitability Factor (RMW):** The fund favors highly profitable firms.")
            else:
                interpretations.append(f"**Profitability Factor (RMW):** The fund favors low profitability firms.")
        else:
            interpretations.append(f"**Profitability Factor (RMW):** The fund shows no significant exposure to profitability factors.")
        
        # Investment factor (CMA)
        cma_coef = model.params['CMA']
        cma_pval = model.pvalues['CMA']
        if cma_pval < 0.1:
            if cma_coef > 0:
                interpretations.append(f"**Investment Factor (CMA):** The fund favors conservatively investing firms.")
            else:
                interpretations.append(f"**Investment Factor (CMA):** The fund favors aggressively investing firms.")
        else:
            interpretations.append(f"**Investment Factor (CMA):** The fund shows no significant investment pattern tilt.")
        
        # Momentum factor (WML)
        wml_coef = model.params['WML']
        wml_pval = model.pvalues['WML']
        if wml_pval < 0.1:
            if wml_coef > 0:
                interpretations.append(f"**Momentum Factor (WML):** The fund has a momentum tilt.")
            else:
                interpretations.append(f"**Momentum Factor (WML):** The fund has a contrarian tilt.")
        else:
            interpretations.append(f"**Momentum Factor (WML):** The fund shows no significant momentum or contrarian tilt.")
        
        for interp in interpretations:
            st.markdown(f"• {interp}")

        # Actual vs Predicted Chart
        st.subheader("Actual vs Predicted Excess Returns")
        
        # Calculate predictions
        regression_data_scaled['predicted'] = model.predict()
        
        # Convert back to percentage points for display
        regression_data_scaled['excess_return_pct'] = regression_data_scaled['excess_return'] * 100
        regression_data_scaled['predicted_pct'] = regression_data_scaled['predicted'] * 100
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=regression_data_scaled.index,
            y=regression_data_scaled['excess_return_pct'],
            mode='lines',
            name='Actual Excess Return',
            line=dict(color='blue', width=2)
        ))
        fig_perf.add_trace(go.Scatter(
            x=regression_data_scaled.index,
            y=regression_data_scaled['predicted_pct'],
            mode='lines',
            name='Predicted',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig_perf.update_layout(
            xaxis_title="Date",
            yaxis_title="Excess Return (%)",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)

        # Residuals Plot
        st.subheader("Residuals Plot")
        
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=regression_data_scaled.index,
            y=model.resid,
            mode='lines+markers',
            name='Residuals',
            line=dict(color='red', width=1),
            marker=dict(size=4)
        ))
        
        fig_resid.update_layout(
            xaxis_title="Date",
            yaxis_title="Residuals",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_resid, use_container_width=True)

        # Detailed Regression Results
        with st.expander("📋 Detailed Regression Results"):
            st.text(str(model.summary()))
        
    except Exception as e:
        st.error(f"Error in factor analysis: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
else:
    st.info("👈 Select a fund from the sidebar to begin analysis")