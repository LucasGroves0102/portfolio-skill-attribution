Portfolio Skill Attribution (PCAT)

Purpose: 
Builds a PCAT and forcasting pipline, ingests market and portfolio data, calculates factor exposures, test for timing skill, then generates reports and visuals 




Tool and Libraries: 
  -Python
  -pandas, numpy, scipy
  -statsmodels
  -scikit-learn
  -yfinance
  -plotly
  -tabulate


Features:
- Data fetching
    1) Pulls stock history from Yahoo Finance
    2) Gets Farma-French 5 factor + momentum from Dartmouth data library
- Portfolio Constrution
    1) Creates equal-weight portfolios or reads from positions.csv
    2) Calculates daily portfolio returns
- Factor Attribution
    1) Models portfolio performance using a six-factor regression framework: MKT, SMB, HML, RMW, CMA, MOM
    2) Estimates alpha and beta with Newey-West HAC errors
    3) Saves t-statistics and R^2
- Timing Tests
    1) implemented Treynor-Mazuy and Henriksson-Merton market timing regressions
- Skill Metrics
    1) Computes Sharpe, Information ration, CGAR, Max Drawdown
    2) Generates CSV
- Forecasting
    1) Factor menan, EWMA, or VAR(1) forecasts
    2) Produces expected excess returns and factor shock scenarios
- Visuals
    1) Growth of $1 (Portfolio vs Market)
    2) Rolling 126-day factor betas

Project Structure:
 - data/ : Input/output CSV (Prices, factors, attribution, forecasts, etc
 - reports/ : Generated reports + charts
 - src / : source code
     - fetch_data : Download stock prices and Fama-French Factors
     - quick_positions: Builds equal weight positions
     - portfolio_returns: Calculates portfolio daily returns
     - attribtuion: Run factor regressions
     - skill_metrics: Compute Sharpe, IR, CAGR, drawdown
     - forecast: Factor-based forecasts and scenario analysis
     - visuals: Generate charts
     - main: Pipelin entrypoint

Future Additions:
  -Machine Learning Factor Forecasts: Replace OLS with Ridge/Lasso regression to shrink noisy betas, Gradient Boosting, scikit-learn pipelin for feature scaling
  -Additional Factor Models: quality, volatility, liquidity
  -Monte Carlo simulation of portfolio paths
  -Add portfolio uploads and instant attribution reports via web UI
  -connects to API (Bloomberg) for daily refresh
   
  
  
       
