import json


# Read market JSON configuration file
with open('../config/market_config.json', 'r') as market_config:
    # Stock ticker symbols at Euronext Paris
    market_config_data: dict = json.load(market_config)
    target_period: str = market_config_data['Period']  # Possible values: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    market_symbol: str = market_config_data['MarketIndexSymbol']  # Ticker symbol for market index
    target_symbols: list[str] = market_config_data['StockTickerSymbols']  # All ticker symbols for stocks

# Read market JSON configuration file
with open('../config/portfolio_config.json', 'r') as portfolio_config:
    # Stock ticker symbols at Euronext Paris
    portfolio_config_data: dict = json.load(portfolio_config)
    n_assets: int = portfolio_config_data['NumberOfAssets']
    capital: int | float = portfolio_config_data['Capital']
    risk_free_rate: float = portfolio_config_data["RiskFreeRate"]
