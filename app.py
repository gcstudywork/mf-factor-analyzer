import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mfapi import MFAPI
from indiafactorlibrary import IndiaFactorLibrary

# Import comparison utilities
from utils.comparison_utils import (
    calculate_rolling_returns, 
    calculate_return_distribution, 
    get_fund_data_with_validation,
    create_rolling_returns_chart,
    create_distribution_chart,
    calculate_summary_stats,
    generate_insights
)

# Page config
st.set_page_config(
    page_title="MF Return Decomposition Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize
mf = MFAPI()
ifl = IndiaFactorLibrary()

# Load factor data
@st.cache_data
def load_factor_data():
    ff6 = ifl.read('ff6')[0]
    ff6.index = pd.to_datetime(ff6.index)
    return ff6

factor_data = load_factor_data()




# Title
st.title("MF Return Decomposition & Comparison Analyzer")

# Add tabs for different analysis types
tab1, tab2 = st.tabs(["🔍 Single Fund Analysis", "⚖️ Fund Comparison"])

with tab1:


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

                        # Date range selection
                        st.divider()
                        st.subheader("Analysis Period")

                        # Get available data range
                        nav_data = mf.get_scheme_historical_nav(scheme_code)
                        if nav_data:
                            nav_df = pd.DataFrame(nav_data)
                            nav_df['date'] = pd.to_datetime(nav_df['date'], format='%d-%m-%Y')

                            min_date = nav_df['date'].min().date()
                            max_date = nav_df['date'].max().date()

                            col1, col2 = st.columns(2)
                            with col1:
                                start_date = st.date_input(
                                    "Start Date",
                                    value=max(min_date, (max_date - timedelta(days=1095))),
                                    min_value=min_date,
                                    max_value=max_date
                                )
                            with col2:
                                end_date = st.date_input(
                                    "End Date",
                                    value=max_date,
                                    min_value=min_date,
                                    max_value=max_date
                                )

                            # Validate date range
                            days_diff = (end_date - start_date).days
                            months_diff = days_diff / 30.44

                            if days_diff < 0:
                                st.error("End date must be after start date")
                            elif months_diff < 36:
                                st.warning(f"⚠️ Only {months_diff:.1f} months of data. Minimum 36 months recommended for reliable analysis.")
                            else:
                                st.success(f"✅ {months_diff:.1f} months of data selected")

                            # Analyze button
                            if st.button("Run Factor Analysis", type="primary", use_container_width=True):
                                st.session_state['analyze'] = True
                                st.session_state['start_date'] = start_date
                                st.session_state['end_date'] = end_date
                                st.session_state['scheme_code'] = scheme_code
                                st.session_state['scheme_name'] = selected_name
            else:
                st.warning("No funds found. Try different search term.")
        else:
            st.error("Failed to load schemes")

    # Main area
    if 'analyze' in st.session_state and st.session_state['analyze']:
        # Perform analysis
        with st.spinner("Performing factor analysis..."):
            try:
                # Get fund data
                nav_data = mf.get_scheme_historical_nav(
                    st.session_state['scheme_code'],
                    from_date=st.session_state['start_date'].strftime('%Y-%m-%d'),
                    to_date=st.session_state['end_date'].strftime('%Y-%m-%d')
                )
                
                fund_df = pd.DataFrame(nav_data)
                fund_df['date'] = pd.to_datetime(fund_df['date'], format='%d-%m-%Y')
                fund_df['nav'] = pd.to_numeric(fund_df['nav'], errors='coerce')
                fund_df = fund_df.dropna().sort_values('date')
                fund_df = fund_df.set_index('date')
                
                # Resample to monthly
                fund_monthly = fund_df.resample('M').last()
                fund_monthly['returns'] = fund_monthly['nav'].pct_change() * 100
                fund_monthly = fund_monthly.dropna()
                
                # Merge with factor data
                merged_data = fund_monthly.merge(factor_data, left_index=True, right_index=True, how='inner')
                
                if len(merged_data) < 12:
                    st.error("Insufficient overlapping data with factor library. Please select a different date range.")
                else:
                    # Calculate excess returns
                    merged_data['excess_return'] = merged_data['returns'] - merged_data['RF']
                    
                    # Prepare regression
                    factors = ['MF', 'SMB5', 'HML', 'RMW', 'CMA', 'WML']
                    X = merged_data[factors]
                    X = sm.add_constant(X)
                    y = merged_data['excess_return']
                    
                    # Run regression
                    model = sm.OLS(y, X).fit()
                    
                    # Display results  
                    st.markdown("## Summary Statistics")

                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.markdown("**Factor Model**")
                        st.markdown("FF6 + Momentum")
                    with col2:
                        st.markdown("**Period**")
                        st.markdown(f"{st.session_state['start_date']} to {st.session_state['end_date']}")
                    with col3:
                        st.markdown("**Observations**")
                        st.markdown(f"{len(merged_data)}")
                    with col4:
                        st.markdown("**R-squared**")
                        st.markdown(f"{model.rsquared:.3f}")
                    with col5:
                        st.markdown("**Adj R-squared**")
                        st.markdown(f"{model.rsquared_adj:.3f}")

                    st.divider()
                    # Factor exposures chart
                    st.header("Factor Exposures")

                    coefficients = model.params[1:]  # Exclude intercept
                    pvalues = model.pvalues[1:]

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

                    # Create bar chart
                    factor_names = ['const<br>(Intercept)', 'MF<br>(Market)', 'SMB<br>(Size)',
                                'HML<br>(Value)', 'RMW<br>(Profitability)',
                                'CMA<br>(Investment)', 'WML<br>(Momentum)']

                    all_coeffs = [model.params['const']] + coefficients.tolist()
                    all_pvals = [model.pvalues['const']] + pvalues.tolist()
                    all_sigs = ['***' if model.pvalues['const'] < 0.01 else
                            '**' if model.pvalues['const'] < 0.05 else
                            '*' if model.pvalues['const'] < 0.1 else ''] + sig_markers

                    colors = ['lightblue' if p < 0.05 else 'lightcoral' for p in all_pvals]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=factor_names,
                        y=all_coeffs,
                        marker_color=colors,
                        text=[f"<b>{coef:.2f} {sig}</b>" for coef, sig in zip(all_coeffs, all_sigs)],
                        textposition='outside',
                        textfont=dict(size=14, family="Arial, sans-serif"),
                    ))

                    fig.update_layout(
                        title=dict(
                            text="Factor Exposures",
                            font=dict(size=20, family="Arial, sans-serif", color="white")
                        ),
                        xaxis=dict(
                            title="Factor",
                            title_font=dict(size=16, family="Arial, sans-serif"),
                            tickfont=dict(size=14, family="Arial, sans-serif")
                        ),
                        yaxis=dict(
                            title="Coefficient",
                            title_font=dict(size=16, family="Arial, sans-serif"),
                            tickfont=dict(size=14, family="Arial, sans-serif")
                        ),
                        showlegend=False,
                        height=500,
                        hovermode='x',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=14)
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                    # Add significance note
                    fig.add_annotation(
                        text="<i>Significance levels: * p<0.1, ** p<0.05, *** p<0.01</i>",
                        xref="paper", yref="paper",
                        x=0, y=-0.25,
                        showarrow=False,
                        font=dict(size=12, family="Arial, sans-serif", color="gray")
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()

                    # Interpretation
                    st.header("Interpretation")

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

                    # Market factor
                    mf_coef = model.params['MF']
                    if mf_coef > 1.2:
                        interpretations.append(f"**Market:** The fund has high sensitivity to how the broader market moves.")
                    elif mf_coef > 0.8:
                        interpretations.append(f"**Market:** The fund moves in line with the broader market.")
                    else:
                        interpretations.append(f"**Market:** The fund has low sensitivity to how the broader market moves.")

                    # Size factor
                    if model.pvalues['SMB5'] < 0.1:
                        if model.params['SMB5'] > 0:
                            interpretations.append(f"**Size Factor (SMB):** The fund favors small-cap stocks—more exposure to firms with smaller market capitalization.")
                        else:
                            interpretations.append(f"**Size Factor (SMB):** The fund favors large-cap stocks—more exposure to firms with larger market capitalization.")

                    # Value factor
                    if model.pvalues['HML'] < 0.1:
                        if model.params['HML'] > 0:
                            interpretations.append(f"**Value Factor (HML):** The fund tilts toward value stocks—typically those with high book-to-market ratios.")
                        else:
                            interpretations.append(f"**Value Factor (HML):** The fund tilts toward growth stocks—typically those with low book-to-market ratios and high price multiples.")

                    # Profitability factor
                    if model.pvalues['RMW'] < 0.1:
                        if model.params['RMW'] > 0:
                            interpretations.append(f"**Profitability Factor (RMW):** The fund is exposed to highly profitable firms.")
                        else:
                            interpretations.append(f"**Profitability Factor (RMW):** The fund is exposed to low profitability or unprofitable firms.")

                    # Investment factor
                    if model.pvalues['CMA'] < 0.1:
                        if model.params['CMA'] > 0:
                            interpretations.append(f"**Investment Factor (CMA):** The fund tilts toward conservatively investing firms—those with low investment rates.")
                        else:
                            interpretations.append(f"**Investment Factor (CMA):** The fund tilts toward aggressively investing firms—those that reinvest heavily in expansion.")

                    # Momentum factor
                    if model.pvalues['WML'] < 0.1:
                        if model.params['WML'] > 0:
                            interpretations.append(f"**Momentum Factor (WML):** The fund has a momentum tilt, favoring stocks that have performed well recently.")
                        else:
                            interpretations.append(f"**Momentum Factor (WML):** The fund has a contrarian tilt, favoring stocks that have performed poorly recently.")

                    for interp in interpretations:
                        st.markdown(f"• {interp}")

                    st.divider()

                    # Actual vs Predicted
                    st.header("Actual vs Predicted Excess Returns")

                    merged_data['predicted'] = model.predict(X)

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=merged_data.index,
                        y=merged_data['excess_return'],
                        mode='lines',
                        name='Actual Excess Return',
                        line=dict(color='blue', width=2)
                    ))
                    fig2.add_trace(go.Scatter(
                        x=merged_data.index,
                        y=merged_data['predicted'],
                        mode='lines',
                        name='Predicted',
                        line=dict(color='orange', width=2, dash='dash')
                    ))

                    fig2.update_layout(
                        title=dict(
                            text="Actual vs Predicted Excess Returns",
                            font=dict(size=20, family="Arial, sans-serif")
                        ),
                        xaxis=dict(
                            title='Date',
                            title_font=dict(size=16, family="Arial, sans-serif"),
                            tickfont=dict(size=12, family="Arial, sans-serif")
                        ),
                        yaxis=dict(
                            title='Excess Return (%)',
                            title_font=dict(size=16, family="Arial, sans-serif"),
                            tickfont=dict(size=12, family="Arial, sans-serif")
                        ),
                        height=400,
                        hovermode='x unified',
                        font=dict(family="Arial, sans-serif", size=14),
                        legend=dict(font=dict(size=14))
                    )

                    st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Error performing analysis: {str(e)}")
                st.exception(e)
    else:
        st.info("👈 Select a fund from the sidebar to begin analysis")


with tab2:
    # New comparison functionality
    st.header("Fund Comparison Analysis")
    
    # Sidebar for fund comparison
    with st.sidebar:
        st.header("Fund Comparison Setup")
        
        # Initialize comparison funds in session state
        if 'comparison_funds' not in st.session_state:
            st.session_state['comparison_funds'] = []
        
        # Get schemes for comparison
        if scheme_codes:
            # Search and add funds
            search_term_comp = st.text_input("Search fund name", "", key="comp_search")
            
            if search_term_comp:
                filtered_names_comp = [name for name in scheme_names.keys() 
                                     if search_term_comp.lower() in name.lower()]
            else:
                filtered_names_comp = list(scheme_names.keys())[:100]
            
            if filtered_names_comp:
                selected_fund_comp = st.selectbox(
                    "Select Fund to Add",
                    options=filtered_names_comp,
                    key="comp_fund_select"
                )
                
                if st.button("Add Fund", key="add_fund_btn"):
                    if selected_fund_comp not in [fund['name'] for fund in st.session_state['comparison_funds']]:
                        fund_info = {
                            'name': selected_fund_comp,
                            'code': scheme_names[selected_fund_comp]
                        }
                        st.session_state['comparison_funds'].append(fund_info)
                        st.success(f"Added {selected_fund_comp}")
                        st.rerun()
                    else:
                        st.warning("Fund already added")
            
            # Display selected funds
            if st.session_state['comparison_funds']:
                st.subheader("Selected Funds")
                for i, fund in enumerate(st.session_state['comparison_funds']):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{i+1}. {fund['name'][:50]}...")
                    with col2:
                        if st.button("❌", key=f"remove_{i}"):
                            st.session_state['comparison_funds'].pop(i)
                            st.rerun()
                
                # Date range for comparison
                st.divider()
                st.subheader("Comparison Period")
                
                comp_start_date = st.date_input(
                    "Start Date",
                    value=datetime.now().date() - timedelta(days=1825),  # 5 years
                    key="comp_start_date"
                )
                comp_end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date(),
                    key="comp_end_date"
                )
                
                if st.button("Compare Funds", type="primary", use_container_width=True, key="compare_btn"):
                    st.session_state['run_comparison'] = True
                    st.session_state['comp_start_date'] = comp_start_date
                    st.session_state['comp_end_date'] = comp_end_date

    # Comparison results
    if 'run_comparison' in st.session_state and st.session_state['run_comparison'] and st.session_state['comparison_funds']:
        
        with st.spinner("Analyzing funds for comparison..."):
            # Collect data for all funds
            fund_data = {}
            fund_availabilities = {}
            
            for fund in st.session_state['comparison_funds']:
                nav_df, availability = get_fund_data_with_validation(
                    mf,
                    fund['code'], 
                    st.session_state['comp_start_date'], 
                    st.session_state['comp_end_date']
                )
                
                if nav_df is not None:
                    fund_data[fund['name']] = nav_df
                    fund_availabilities[fund['name']] = availability
                else:
                    st.warning(f"Could not load data for {fund['name']}: {availability}")
            
            if fund_data:
                # 1. Rolling Returns Comparison
                st.header("🔄 Rolling Returns Comparison")
                
                # Show data availability
                st.subheader("Data Availability Check")
                availability_df = pd.DataFrame(fund_availabilities).T
                availability_df = availability_df.replace({True: "✅", False: "❌"})
                st.dataframe(availability_df, use_container_width=True)
                
                # Calculate rolling returns for each fund
                all_rolling_data = {}
                for fund_name, nav_df in fund_data.items():
                    rolling_data = calculate_rolling_returns(nav_df)
                    all_rolling_data[fund_name] = rolling_data
                
                # Create rolling returns charts
                periods_to_show = ['1Y', '2Y', '3Y', '5Y']
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                
                for period in periods_to_show:
                    # Check which funds have data for this period
                    funds_with_data = [name for name, data in all_rolling_data.items() if period in data]
                    
                    if funds_with_data:
                        st.subheader(f"{period} Rolling Returns")
                        fig = create_rolling_returns_chart(all_rolling_data, period, colors)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # 2. Return Distribution Comparison
                st.header("📊 Return Distribution Analysis")
                
                # Calculate distributions for all funds
                all_distributions = {}
                for fund_name, nav_df in fund_data.items():
                    distribution = calculate_return_distribution(nav_df)
                    all_distributions[fund_name] = distribution
                
                # Create distribution comparison chart
                fig_dist = create_distribution_chart(all_distributions, colors)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Summary statistics table
                st.subheader("📈 Summary Statistics")
                
                summary_stats = calculate_summary_stats(fund_data)
                summary_df = pd.DataFrame(summary_stats).T
                summary_df = summary_df.round(2)
                st.dataframe(summary_df, use_container_width=True)
                
                # Key insights
                st.subheader("🔍 Key Insights")
                insights = generate_insights(summary_stats)
                
                for insight in insights:
                    st.markdown(f"• {insight}")
                
            else:
                st.error("No valid fund data available for comparison")
    
    elif not st.session_state.get('comparison_funds'):
        st.info("👈 Add funds from the sidebar to start comparison")
    
    else:
        st.info("Select funds and click 'Compare Funds' to begin analysis")
