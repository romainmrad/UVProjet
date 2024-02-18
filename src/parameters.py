import os
import json
from typing import Literal

config_dir_path = ''
if os.path.exists('../config/'):
    config_dir_path = '../config/'
elif os.path.exists('./config/'):
    config_dir_path = './config/'
else:
    raise NotADirectoryError('Please provide a valid directory path for config files')

# Read market JSON configuration file
with open(config_dir_path + 'market_config.json', 'r') as market_config:
    market_config_data: dict = json.load(market_config)
    target_period: str = market_config_data['period']  # Possible values: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    market_symbol: str = market_config_data['marketIndexSymbol']  # Ticker symbol for market index
    target_symbols: list[str] = market_config_data['stockTickerSymbols']  # All ticker symbols for stocks

# Read market JSON configuration file
with open(config_dir_path + 'portfolio_config.json', 'r') as portfolio_config:
    portfolio_config_data: dict = json.load(portfolio_config)
    n_assets: int = portfolio_config_data['numberOfAssets']  # Number of portfolio assets
    capital: int | float = portfolio_config_data['capital']  # Invested capital
    risk_free_rate: float = portfolio_config_data['riskFreeRate']  # Risk-free rate
    optimisation_factor: Literal['sharpeRatio', 'risk', 'return'] = portfolio_config_data['optimisationFactor']
    if optimisation_factor not in ['sharpeRatio', 'risk', 'return']:
        raise ValueError(f'{optimisation_factor} is not a valid optimisation factor')
    minimum_share_price: int | float = portfolio_config_data['minimumSharePrice']
