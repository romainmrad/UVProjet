import json
import os
import yfinance as yf


def load_current_portfolio_data() -> dict:
    """
    Load current portfolio from JSON file
    :return: Dictionary of current portfolio
    """
    # Checking if file path exists
    if os.path.exists('../data/portfolio/current_portfolio.json'):
        portfolio_file_path = '../data/portfolio/current_portfolio.json'  # Path for src scope
    elif os.path.exists('./data/portfolio/current_portfolio.json'):
        portfolio_file_path = './data/portfolio/current_portfolio.json'  # Path for project scope
    else:
        raise NotADirectoryError('Please provide a valid directory path for config files')
    # Loading data to dictionary and returning it
    with open(portfolio_file_path, mode='r') as f:
        portfolio_data = json.load(f)
    return portfolio_data


def estimate_stock_movement(stock_data) -> float:
    pass


def estimate_bearish_stocks(portfolio_data: dict):
    """
    Estimate bearish stocks in current portfolio
    :param portfolio_data: dictionary containing current portfolio data
    :return: list of ticker symbols for bearish stocks in current portfolio
    """
    stock_data = {}
    bearish_stocks = []
    for ticker in portfolio_data['stocks'].keys():
        stock_data[ticker] = {}
        stock_data[ticker]['data'] = yf.download(tickers=ticker, period='5y')['Adj Close'].pct_change(periods=-1)[:-1]
        stock_data[ticker]['data'] = stock_data[ticker]['data'].fillna(method='interpolate', axis=0)
        stock_data[ticker]['estimatedPrice'] = estimate_stock_movement(stock_data[ticker]['data'])
        if stock_data[ticker]['estimatedPrice'] > stock_data[ticker]['data'][-1]:
            stock_data[ticker]['movementType'] = 'bull'
        elif stock_data[ticker]['estimatedPrice'] - stock_data[ticker]['data'][-1] <= 5:
            stock_data[ticker]['movementType'] = 'bear'
            bearish_stocks.append(ticker)
    return bearish_stocks
