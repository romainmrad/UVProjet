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
    # market_index_symbol: str = market_config_data['marketIndexSymbol']  # Ticker symbol for market index
    target_symbols: list[str] = market_config_data['stockTickerSymbols']  # All ticker symbols for stocks
    risk_free_rate: float = market_config_data['riskFreeRate']  # Risk-free rate
    minimum_share_price: int | float = market_config_data['minimumSharePrice']  # Minmum share price
    trading_fee: int | float = market_config_data['tradingFee']  # Trading fee

# Read portfolio JSON configuration file
with open(config_dir_path + 'portfolio_config.json', 'r') as portfolio_config:
    portfolio_config_data: dict = json.load(portfolio_config)
    number_of_assets: int = portfolio_config_data['numberOfAssets']  # Number of portfolio assets
    capital: int | float = portfolio_config_data['capital']  # Invested capital
    optimisation_factor: Literal[
        'sharpeRatio',
        'risk',
        'return'
    ] = portfolio_config_data['optimisationFactor']
    if optimisation_factor not in ['sharpeRatio', 'risk', 'return']:
        raise ValueError(f'{optimisation_factor} is not a valid optimisation factor')
    # Minimum proportion of portfolio in one company
    minimum_proportion: float = portfolio_config_data['minimumProportion']
    # Initial stocks choice method
    initialisation_method: Literal[
        'prediction',
        'historical'
    ] = portfolio_config_data['initialisationMethod']

# Read prediction JSON configuration file
with open(config_dir_path + 'prediction_config.json', 'r') as prediction_config:
    prediction_config_data: dict = json.load(prediction_config)
    # The prediction metric
    prediction_metric: Literal[
        'r2_score',
        'direction',
        'mean_squared_error'
    ] = prediction_config_data['predictionMetric']
    # Maximum number of layers to try
    max_layers: int = prediction_config_data['maxLayers']
    # Lookback for prediction
    lookback: int = prediction_config_data['lookback']
