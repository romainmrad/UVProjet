import json


# Read JSON configuration file
with open('../config/market_config.json', 'r') as market_config:
    # Stock ticker symbols at Euronext Paris
    market_config_data = json.load(market_config)
    target_period: str = market_config_data['Period']  # Possible values: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    market_symbol: str = market_config_data['MarketIndexSymbol']  # Ticker symbol for market index
    target_symbols: list[str] = market_config_data['MarketTickerSymbols']  # All ticker symbols for stocks
