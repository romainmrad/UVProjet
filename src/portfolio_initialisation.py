import json
import os
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from src.parameters import target_symbols, target_period, n_assets, capital, risk_free_rate, optimisation_factor
from src.grapher import plot_portfolio_visualisation
from src.tools import get_company_name


def compute_optimisation_factor(
        x: np.ndarray,
        stock_returns: pd.DataFrame,
        factor: str,
        rfr: float
) -> float:
    """
    Compute negative Sharpe ratio for a portfolio of weights x
    :param x: the weights of the portfolio
    :param stock_returns: dataframe of stock daily returns
    :param factor: the factor to optimise. Values are 'SharpeRatio', 'Risk' and 'Return'
    :param rfr: risk-free rate for ratio computing
    :return: negative sharpe ratio
    """
    weighted_returns = (stock_returns * x).sum(axis=1)
    match factor:
        case 'SharpeRatio':
            expected_return = weighted_returns.mean()
            risk = weighted_returns.var()
            return - (expected_return - rfr) / risk
        case 'Risk':
            return weighted_returns.var()
        case 'Return':
            return - weighted_returns.mean()
        case _:
            raise ValueError(f'{factor} is not a valid optimisation factor')


def fetch_stocks_returns(
        stock_ticker_symbols: list[str],
        period: str
) -> pd.DataFrame:
    """
    Use Yahoo Finance API to fetch all Adjusted Close price data
    :param stock_ticker_symbols: list of target stock ticker symbols
    :param period: historical time period to load
    :return: Dataframe containing historical adjusted close price for all ticker symbols
    """
    stock_data = yf.download(stock_ticker_symbols, period=period)['Adj Close']
    stock_data.sort_values(by='Date', ascending=False, inplace=True)
    return stock_data.pct_change(periods=-1)[:-1].dropna(axis=1)


def extract_top_n_stocks(
        n: int,
        stock_ticker_symbols: list[str],
        period: str
) -> pd.DataFrame:
    """
    Extract top n performing stocks based on their return
    :param n: number of stocks to return
    :param stock_ticker_symbols: list of target stock ticker symbols
    :param period: historical time period to load
    :return: Dataframe containing top n performing stocks and their historical adjusted close price
    """
    returns = fetch_stocks_returns(stock_ticker_symbols=stock_ticker_symbols, period=period)
    mean_returns = list(returns.mean())
    top_indices = np.argsort(mean_returns)[-n:][::-1]
    return returns.iloc[:, top_indices]


def format_portfolio_data(
        n: int,
        invested_capital: int | float,
        stock_ticker_symbols: list[str],
        weights: np.ndarray,
        stock_returns: pd.DataFrame,
) -> dict:
    """
    Format portfolio data for JSON file
    :param n: number of assets
    :param invested_capital: invested capital
    :param stock_ticker_symbols: ticker symbols of portfolio stock
    :param weights: individual stock weights
    :param stock_returns: daily returns for portfolio stocks
    :return: dictionary-formatted portfolio data
    """
    # Compute weighted returns
    weighted_returns = (stock_returns * weights).sum(axis=1)
    expected_return = weighted_returns.mean()
    risk = weighted_returns.var()
    # Initialise portfolio dictionary
    portfolio = {
        'Characteristics': {
            'NumberOfAssets': n,
            'ExpectedReturn': expected_return,
            'Risk': risk,
            'SharpeRatio': (expected_return - risk_free_rate) / risk,
        },
        'Stocks': {}
    }

    # Iterate over portfolio stocks
    for symbol, weight in zip(stock_ticker_symbols, weights):
        portfolio['Stocks'][symbol] = {}
        # Add company name
        portfolio['Stocks'][symbol]['CompanyName'] = get_company_name(symbol)
        # Add weight
        portfolio['Stocks'][symbol]['Weight'] = weight
        # Add amount invested
        portfolio['Stocks'][symbol]['Amount'] = weight * invested_capital
        # Fetch current closing price
        current_closing_price = yf.download(symbol, period='1D')['Adj Close'].values[0]
        # Compute number of shares
        portfolio['Stocks'][symbol]['Shares'] = int((weight * invested_capital) // current_closing_price)
        # Adding share price
        portfolio['Stocks'][symbol]['PricePerShare'] = current_closing_price
    return portfolio


def compute_optimal_portfolio(
        n: int,
        invested_capital: int | float,
        stock_ticker_symbols: list[str],
        period: str
) -> None:
    """
    Compute optimal initial portfolio
    :param n: number of assets
    :param invested_capital: invested capital
    :param stock_ticker_symbols: ticker symbols of portfolio stock
    :param period: historical time period to load
    """
    # Fetch top n stock data
    returns = extract_top_n_stocks(n=n, stock_ticker_symbols=stock_ticker_symbols, period=period)
    portfolio_symbols = list(returns.columns)
    # Set up the optimisation problem
    weights = 1/n * np.ones(n)
    weight_bounds = Bounds(lb=0.1 * np.ones(n), ub=np.ones(1))
    linear_constraint = LinearConstraint(np.ones(n), lb=1, ub=1)
    # Solve optimisation problem
    optimisation = minimize(
        fun=compute_optimisation_factor,
        x0=weights,
        method='trust-constr',
        constraints=linear_constraint,
        args=(returns, optimisation_factor, risk_free_rate),
        bounds=weight_bounds
    )
    optimal_weights = optimisation.x

    # Format portfolio as JSON
    optimal_portfolio = format_portfolio_data(
        n=n,
        invested_capital=invested_capital,
        stock_ticker_symbols=portfolio_symbols,
        weights=optimal_weights,
        stock_returns=returns,
    )
    # Check if the path exists
    if not os.path.exists('../data/portfolio/'):
        # Create the directory
        os.makedirs('../data/portfolio/')
    # Output data to JSON
    with open('../data/portfolio/current_portfolio.json', 'w') as file:
        json.dump(optimal_portfolio, file, indent=4)
    # Visualise portfolio
    plot_portfolio_visualisation(optimal_portfolio)
    print('Successful output of portfolio data to JSON')
    print('Successful plotted of portfolio diversification')


if __name__ == '__main__':
    compute_optimal_portfolio(
        n=n_assets,
        invested_capital=capital,
        stock_ticker_symbols=target_symbols,
        period=target_period
    )
