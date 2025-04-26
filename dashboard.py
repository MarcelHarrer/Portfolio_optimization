import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from functools import lru_cache
import os

# Define functions for portfolio optimization
def get_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data for given tickers.
    Skips tickers that fail to fetch data and logs a warning.
    """
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            if 'Adj Close' in stock_data.columns:
                data[ticker] = stock_data['Adj Close']
            else:
                print(f"Warning: 'Adj Close' column not found for ticker {ticker}. Skipping.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}. Skipping.")
    if data.empty:
        raise ValueError("No valid data fetched for the selected tickers.")
    return data

@lru_cache(maxsize=10)
def get_stock_data_cached(tickers, start_date, end_date):
    return get_stock_data(tickers, start_date, end_date)

def calculate_portfolio_metrics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    portfolio_return, portfolio_volatility = calculate_portfolio_metrics(weights, returns)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def optimize_portfolio(returns, risk_free_rate):
    num_assets = returns.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1/num_assets] * num_assets)
    result = minimize(negative_sharpe_ratio, 
                      initial_weights,
                      args=(returns, risk_free_rate),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return result.x

# Create Dash app
app = Dash(__name__)
app.title = "Portfolio Optimization Dashboard"  # Set the browser tab title

# Layout with dropdown for stock selection, button, and loading spinner
app.layout = html.Div([
    html.Div([
        html.H1("Interactive Portfolio Optimization Dashboard", style={'textAlign': 'center', 'color': '#4CAF50'}),
        html.Div([
            html.Label("Select Stocks:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='stock-picker',
                options=[
                    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
                    {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                    {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
                    {'label': 'Meta (META)', 'value': 'META'},
                    {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
                    {'label': 'NVIDIA (NVDA)', 'value': 'NVDA'},
                    {'label': 'Netflix (NFLX)', 'value': 'NFLX'},
                    {'label': 'Adobe (ADBE)', 'value': 'ADBE'},
                    {'label': 'Intel (INTC)', 'value': 'INTC'},
                    {'label': 'Cisco (CSCO)', 'value': 'CSCO'},
                    {'label': 'PepsiCo (PEP)', 'value': 'PEP'},
                    {'label': 'Coca-Cola (KO)', 'value': 'KO'},
                    {'label': 'Procter & Gamble (PG)', 'value': 'PG'},
                    {'label': 'Johnson & Johnson (JNJ)', 'value': 'JNJ'}
                ],
                value=['AAPL', 'MSFT', 'GOOGL'],  # Default selected stocks
                multi=True,
                style={'marginBottom': '10px'}
            ),
            html.Label("Select Start Date:", style={'fontWeight': 'bold'}),
            dcc.DatePickerSingle(
                id='start-date-picker',
                date='2020-01-01',  # Default start date
                style={'marginBottom': '10px'}
            ),
            html.Label("Select End Date:", style={'fontWeight': 'bold'}),
            dcc.DatePickerSingle(
                id='end-date-picker',
                date='2024-01-01',  # Default end date
                style={'marginBottom': '10px'}
            ),
            html.Button("Calculate Portfolio", id='calculate-button', n_clicks=0, style={
                'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'padding': '10px 20px',
                'textAlign': 'center', 'textDecoration': 'none', 'display': 'inline-block', 'fontSize': '16px',
                'marginTop': '10px', 'cursor': 'pointer'
            })
        ], style={'width': '50%', 'margin': '0 auto', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': '#f9f9f9'}),
    ], style={'padding': '20px', 'backgroundColor': '#f0f0f0'}),
    dcc.Loading(
        id="loading",
        type="circle",  # You can use "circle", "dot", or "default"
        children=[
            html.Div(id='portfolio-metrics', style={'marginTop': '20px', 'textAlign': 'center'}),
            dcc.Graph(id='efficient-frontier', style={'marginTop': '20px'})
        ]
    )
], style={'fontFamily': 'Arial, sans-serif', 'lineHeight': '1.6', 'backgroundColor': '#ffffff'})

# Callback to calculate portfolio metrics and efficient frontier on button click
@app.callback(
    [Output('portfolio-metrics', 'children'),
     Output('efficient-frontier', 'figure')],
    [Input('calculate-button', 'n_clicks')],
    [State('stock-picker', 'value'),
     State('start-date-picker', 'date'),
     State('end-date-picker', 'date')]
)
def update_dashboard(n_clicks, selected_tickers, start_date, end_date):
    if n_clicks == 0 or not selected_tickers or not start_date or not end_date:
        return "Please select stocks, dates, and click the button to calculate.", {}

    risk_free_rate = 0.02
    try:
        stock_data = get_stock_data_cached(tuple(selected_tickers), start_date, end_date)
    except ValueError as e:
        return str(e), {}

    returns = stock_data.pct_change().dropna()

    # Optimize portfolio
    optimal_weights = optimize_portfolio(returns, risk_free_rate)
    portfolio_return, portfolio_volatility = calculate_portfolio_metrics(optimal_weights, returns)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Generate efficient frontier
    returns_array = []
    volatility_array = []
    optimal_point = None  # To store the optimal portfolio point

    # Calculate the efficient frontier only up to the optimal portfolio's return
    for ret in np.linspace(0, portfolio_return, 50):  # Adjust the number of points as needed
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: calculate_portfolio_metrics(x, returns)[0] - ret}
        )
        bounds = tuple((0, 1) for _ in range(len(selected_tickers)))
        result = minimize(lambda x: calculate_portfolio_metrics(x, returns)[1],
                          [1/len(selected_tickers)] * len(selected_tickers),
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
        if result.success:
            volatility_array.append(result.fun)
            returns_array.append(ret)

            # Check if this point matches the optimal portfolio
            if np.isclose(ret, portfolio_return, atol=1e-4):
                optimal_point = (result.fun, ret)

    # Ensure the optimal point is included
    if optimal_point:
        portfolio_volatility, portfolio_return = optimal_point

    # Portfolio metrics
    metrics = html.Div([
        html.H3("Optimal Portfolio Weights:"),
        html.Ul([html.Li(f"{ticker}: {weight:.4f}".strip()) for ticker, weight in zip(selected_tickers, optimal_weights)]),
        html.H3("Portfolio Metrics:"),
        html.P(f"Expected Return: {portfolio_return:.4f}"),
        html.P(f"Volatility: {portfolio_volatility:.4f}"),
        html.P(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    ])

    # Efficient frontier figure
    figure = {
        'data': [
            go.Scatter(x=volatility_array, y=returns_array, mode='lines', name='Efficient Frontier'),
            go.Scatter(x=[portfolio_volatility], y=[portfolio_return], mode='markers', name='Optimal Portfolio', marker=dict(size=10, color='red'))
        ],
        'layout': go.Layout(
            title='Efficient Frontier and Optimal Portfolio',
            xaxis=dict(title='Volatility'),
            yaxis=dict(title='Return'),
            showlegend=True
        )
    }

    return metrics, figure

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))  # Use environment variable PORT or default to 8080
    app.run(debug=False, port=port)