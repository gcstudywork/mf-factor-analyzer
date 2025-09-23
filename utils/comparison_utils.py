import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def calculate_rolling_returns(nav_data, periods=[12, 24, 36, 60]):
    """Calculate rolling returns for different periods"""
    nav_data = nav_data.copy()
    nav_data['returns'] = nav_data['nav'].pct_change()
    
    rolling_data = {}
    for period in periods:
        if len(nav_data) >= period:
            # Calculate rolling annualized returns
            rolling_returns = nav_data['nav'].pct_change(periods=period).dropna() * 100
            if len(rolling_returns) > 0:
                rolling_data[f'{period//12}Y'] = rolling_returns
    
    return rolling_data

def calculate_return_distribution(nav_data):
    """Calculate return distribution across different buckets"""
    monthly_returns = nav_data['nav'].pct_change().dropna() * 100
    
    # Define buckets
    buckets = [
        ('< -10%', lambda x: x < -10),
        ('-10% to 0%', lambda x: (-10 <= x) & (x < 0)),
        ('0% to 10%', lambda x: (0 <= x) & (x < 10)),
        ('10% to 15%', lambda x: (10 <= x) & (x < 15)),
        ('15% to 20%', lambda x: (15 <= x) & (x < 20)),
        ('20% to 25%', lambda x: (20 <= x) & (x < 25)),
        ('25% to 35%', lambda x: (25 <= x) & (x < 35)),
        ('> 35%', lambda x: x >= 35)
    ]
    
    distribution = {}
    for bucket_name, condition in buckets:
        count = len(monthly_returns[condition(monthly_returns)])
        percentage = (count / len(monthly_returns)) * 100 if len(monthly_returns) > 0 else 0
        distribution[bucket_name] = {
            'count': count,
            'percentage': percentage
        }
    
    return distribution

def get_fund_data_with_validation(mf_client, scheme_code, start_date, end_date):
    """Get fund data with availability validation"""
    try:
        # Handle both string and date objects
        if isinstance(start_date, str):
            start_date_str = start_date
        else:
            start_date_str = start_date.strftime('%Y-%m-%d')
            
        if isinstance(end_date, str):
            end_date_str = end_date
        else:
            end_date_str = end_date.strftime('%Y-%m-%d')
        
        nav_data = mf_client.get_scheme_historical_nav(
            scheme_code,
            from_date=start_date_str,
            to_date=end_date_str
        )
        
        if not nav_data:
            return None, "No data available for selected period"
        
        fund_df = pd.DataFrame(nav_data)
        fund_df['date'] = pd.to_datetime(fund_df['date'], format='%d-%m-%Y')
        fund_df['nav'] = pd.to_numeric(fund_df['nav'], errors='coerce')
        fund_df = fund_df.dropna().sort_values('date')
        fund_df = fund_df.set_index('date')
        
        # Check data availability for different periods
        total_months = len(fund_df)
        availability = {
            '1Y': total_months >= 12,
            '2Y': total_months >= 24,
            '3Y': total_months >= 36,
            '5Y': total_months >= 60
        }
        
        return fund_df, availability
        
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def create_rolling_returns_chart(all_rolling_data, period, colors):
    """Create rolling returns chart for a specific period"""
    funds_with_data = [name for name, data in all_rolling_data.items() if period in data]
    
    if not funds_with_data:
        return None
    
    fig = go.Figure()
    
    for i, fund_name in enumerate(funds_with_data):
        rolling_returns = all_rolling_data[fund_name][period]
        fig.add_trace(go.Scatter(
            x=rolling_returns.index,
            y=rolling_returns.values,
            mode='lines',
            name=fund_name[:30] + "..." if len(fund_name) > 30 else fund_name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title=f"{period} Rolling Returns Comparison",
        xaxis_title="Date",
        yaxis_title=f"{period} Rolling Return (%)",
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_distribution_chart(all_distributions, colors):
    """Create return distribution comparison chart"""
    bucket_names = ['< -10%', '-10% to 0%', '0% to 10%', '10% to 15%', 
                   '15% to 20%', '20% to 25%', '25% to 35%', '> 35%']
    
    fig = go.Figure()
    
    for i, (fund_name, distribution) in enumerate(all_distributions.items()):
        percentages = [distribution[bucket]['percentage'] for bucket in bucket_names]
        
        fig.add_trace(go.Bar(
            name=fund_name[:30] + "..." if len(fund_name) > 30 else fund_name,
            x=bucket_names,
            y=percentages,
            marker_color=colors[i % len(colors)],
            opacity=0.8
        ))
    
    fig.update_layout(
        title="Monthly Return Distribution Comparison",
        xaxis_title="Return Buckets",
        yaxis_title="Percentage of Months (%)",
        barmode='group',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def calculate_summary_stats(fund_data):
    """Calculate summary statistics for all funds"""
    summary_stats = {}
    
    for fund_name, nav_df in fund_data.items():
        monthly_returns = nav_df['nav'].pct_change().dropna() * 100
        
        stats = {
            'Mean Monthly Return (%)': monthly_returns.mean(),
            'Std Dev (%)': monthly_returns.std(),
            'Min Return (%)': monthly_returns.min(),
            'Max Return (%)': monthly_returns.max(),
            'Positive Months (%)': (monthly_returns > 0).sum() / len(monthly_returns) * 100,
            'Sharpe Ratio (Monthly)': monthly_returns.mean() / monthly_returns.std() if monthly_returns.std() > 0 else 0
        }
        summary_stats[fund_name] = stats
    
    return summary_stats

def generate_insights(summary_stats):
    """Generate key insights from summary statistics"""
    insights = []
    
    # Best performing fund
    mean_returns = {name: stats['Mean Monthly Return (%)'] for name, stats in summary_stats.items()}
    best_performer = max(mean_returns, key=mean_returns.get)
    insights.append(f"**Highest Average Return:** {best_performer} ({mean_returns[best_performer]:.2f}% monthly)")
    
    # Most consistent fund
    volatilities = {name: stats['Std Dev (%)'] for name, stats in summary_stats.items()}
    most_consistent = min(volatilities, key=volatilities.get)
    insights.append(f"**Most Consistent:** {most_consistent} ({volatilities[most_consistent]:.2f}% volatility)")
    
    # Best risk-adjusted return
    sharpe_ratios = {name: stats['Sharpe Ratio (Monthly)'] for name, stats in summary_stats.items()}
    best_sharpe = max(sharpe_ratios, key=sharpe_ratios.get)
    insights.append(f"**Best Risk-Adjusted Return:** {best_sharpe} (Sharpe: {sharpe_ratios[best_sharpe]:.3f})")
    
    return insights

def calculate_rolling_return_summary(nav_data, rolling_period_months, fund_name):
    """
    Calculate rolling return statistics and distribution for a specific rolling period
    
    Parameters:
    nav_data: DataFrame with NAV data
    rolling_period_months: int (12, 24, 36, 60 for 1Y, 2Y, 3Y, 5Y)
    fund_name: str
    
    Returns:
    dict with summary statistics and distribution
    """
    # Calculate rolling returns (annualized)
    if len(nav_data) < rolling_period_months:
        return None
    
    # Calculate rolling returns as percentage
    rolling_returns = nav_data['nav'].pct_change(periods=rolling_period_months).dropna()
    
    # Annualize the returns
    years = rolling_period_months / 12
    annualized_returns = ((1 + rolling_returns) ** (1/years) - 1) * 100
    
    if len(annualized_returns) == 0:
        return None
    
    # Calculate statistics
    stats = {
        'Fund Name': fund_name,
        'Average': annualized_returns.mean(),
        'Maximum': annualized_returns.max(),
        'Minimum': annualized_returns.min(),
    }
    
    # Calculate distribution buckets for annualized returns
    total_observations = len(annualized_returns)
    
    distribution = {
        'Less than 0%': (annualized_returns < 0).sum() / total_observations * 100,
        '0 - 10%': ((annualized_returns >= 0) & (annualized_returns < 10)).sum() / total_observations * 100,
        '10 - 20%': ((annualized_returns >= 10) & (annualized_returns < 20)).sum() / total_observations * 100,
        '20 - 30%': ((annualized_returns >= 20) & (annualized_returns < 30)).sum() / total_observations * 100,
        'More than 30%': (annualized_returns >= 30).sum() / total_observations * 100,
    }
    
    # Combine stats and distribution
    result = {**stats, **distribution}
    
    return result

def create_rolling_summary_table(fund_data_dict, rolling_period_months, start_date, end_date):
    """
    Create summary table for multiple funds
    
    Parameters:
    fund_data_dict: dict of {fund_name: nav_dataframe}
    rolling_period_months: int (12, 24, 36, 60)
    start_date, end_date: for display in header
    
    Returns:
    pandas DataFrame
    """
    summaries = []
    
    for fund_name, nav_df in fund_data_dict.items():
        summary = calculate_rolling_return_summary(nav_df, rolling_period_months, fund_name)
        if summary:
            summaries.append(summary)
    
    if not summaries:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(summaries)
    
    # Set Fund Name as index
    df = df.set_index('Fund Name')
    
    # Round to 2 decimal places
    df = df.round(2)
    
    return df
# End of file