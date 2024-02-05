import yfinance as yf
import os
from parameters import target_symbols, target_period


def fetch_stock_historical_data(
        symbol: str,
        period: str
) -> None:
    """
    Use Yahoo Finance API to fetch stock historical data
    :param symbol: stock ticker symbols (i.e. BNP.PA for BNP Paribas)
    :param period: historical time period to load
    :return: Dataframe containing stock historical data
    """
    # Fetching stock data using Yahoo Finance API
    stock_data = yf.Ticker(symbol).history(period=period)[['Open', 'High', 'Low', 'Close']]
    if not stock_data.empty:
        stock_data = stock_data.reset_index()  # Reset index of returned dataframe
        stock_data.dropna(axis=0, inplace=True)  # Drop NA value rows
        print(f'success for {symbol}')  # Log success
        stock_data.to_csv(f'../data/historical/{symbol}.csv')
    # If returned data is empty, it means that the passed ticker symbol is invalid
    else:
        print(f'Invalid ticker symbol: {symbol}')  # Log error


def fetch_market_historical_data(
        symbols: list,
        period: str
) -> None:
    """
    Use Yahoo Finance API to fetch stock historical data
    :param symbols: list of target stock ticker symbols
    :param period: historical time period to load
    :return: None
    """
    # Iterate over historical data directory
    for item in os.listdir('../data/historical/'):
        # Remove unwanted historical data files
        if item.split('.csv')[0] not in symbols:
            os.remove(os.path.join('../data/historical/', item))
    # Fetch historical data for every ticker symbol
    for symbol in symbols:
        fetch_stock_historical_data(symbol=symbol, period=period)


if __name__ == '__main__':
    fetch_market_historical_data(symbols=target_symbols, period=target_period)
