import json
import os
from src.parameters import optimisation_factor


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


def compute_least_performing_stock(portfolio_data: dict) -> str:
    """
    Compute least-performing stock in current portfolio
    :param portfolio_data: dictionary containing current portfolio data
    :return: least-performing stock ticker symbol
    """
    if optimisation_factor == 'SharpeRatio' or optimisation_factor == 'ExpectedReturn':
        # Initialise search
        min_factor_value = float('inf')
        min_factor_symbol = None
        # Iterate over portfolio stocks
        for symbol, stock_data in portfolio_data['Stocks'].items():
            current_value = stock_data[optimisation_factor]
            # Update minimum value
            if current_value < min_factor_value:
                min_factor_value = current_value
                min_factor_symbol = symbol
        # Return least-performing stock ticker symbol
        return min_factor_symbol
    elif optimisation_factor == 'Risk':
        # Initialise search
        max_factor_value = -float('inf')
        max_factor_symbol = None
        # Iterate over portfolio stocks
        for symbol, stock_data in portfolio_data['Stocks'].items():
            current_value = stock_data[optimisation_factor]
            # Update minimum value
            if current_value > max_factor_value:
                max_factor_value = current_value
                max_factor_symbol = symbol
        # Return least-performing stock ticker symbol
        return max_factor_symbol
