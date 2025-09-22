# MF Factor Analyzer - Project Context

## Project Overview
A Streamlit-based mutual fund factor analysis tool using Fama-French 6-factor model for Indian mutual funds.

**Live App**: https://factrd.streamlit.app
**GitHub**: https://github.com/gcstudywork/mf-factor-analyzer

## Tech Stack
- Python 3.11
- Streamlit (web framework)
- MFAPI (Indian mutual fund data)
- India Factor Library (FF6 factors)
- statsmodels (regression)
- Plotly (visualizations)

## Project Structure
mf-factor-analyzer/
├── mfapi/
│   ├── init.py
│   └── client.py          # MFAPI wrapper with caching
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version (3.11)
├── .gitignore
└── README.md

## Key Features Implemented
1. **Fund Search & Selection** - Search 40k+ funds, view details
2. **Date Range Validation** - Warns if < 36 months data
3. **Factor Analysis (FF6)**:
   - Regression with MF, SMB5, HML, RMW, CMA, WML factors
   - R-squared and adjusted R-squared
   - Statistical significance testing (*, **, ***)
   - Factor exposure bar chart
   - Automated interpretation (plain English)
   - Actual vs Predicted returns time series

## Code Architecture

### MFAPI Client (`mfapi/client.py`)
- 1-hour caching for scheme codes
- Methods: `get_scheme_codes()`, `get_scheme_details()`, `get_scheme_historical_nav()`
- Date filtering support

### Main App (`app.py`)
- Session state for analysis persistence
- Monthly resampling of NAV data
- OLS regression with statsmodels
- Plotly charts with custom styling
- Interpretation logic based on coefficients and p-values

## Important Implementation Details

### Date Handling
- Input format: YYYY-MM-DD
- API returns: DD-MM-YYYY
- Monthly resampling: `.resample('M')` (not 'ME' - pandas compatibility)

### Plotly Compatibility
- Use `title_font=` not `titlefont=` (newer Plotly versions)
- Color by significance: blue (<0.05), coral (>=0.05)

### Factor Analysis Flow
1. Fetch NAV data from MFAPI
2. Resample to monthly
3. Calculate returns
4. Merge with FF6 factor data
5. Calculate excess returns (return - RF)
6. Run OLS regression
7. Visualize and interpret

## Deployment Notes
- Streamlit Cloud requires Python 3.11 (specified in runtime.txt)
- No version pins in requirements.txt (auto-resolves compatible versions)
- Auto-deploys on git push to main branch

## Known Issues & Fixes
1. **India Factor Library 406 error**: Upgrade to 0.0.10
2. **Python 3.13 incompatibility**: Use runtime.txt with python-3.11
3. **Plotly titlefont error**: Replace with title_font

## Testing Checklist
- [ ] Search "Nippon India Growth"
- [ ] Select 2020-2024 date range
- [ ] Run analysis
- [ ] Verify charts render
- [ ] Check interpretations make sense

## Future Enhancements (Not Built)
- Export results (CSV/PDF)
- Multiple fund comparison
- Rolling factor analysis
- Portfolio optimization
- Peer comparison
- Risk metrics

## Git Workflow
```bash
git add .
git commit -m "feat: description"
git push  # Auto-deploys to Streamlit Cloud

Key Files Content
requirements.txt
streamlit
pandas
numpy
requests
plotly
statsmodels
scipy
indiafactorlibrary
runtime.txt
python-3.11
Contact & Repository

GitHub User: gcstudywork
Repository: mf-factor-analyzer
Branch: main