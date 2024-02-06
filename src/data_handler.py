import yfinance as yf
import os
from parameters import market_symbol, target_symbols, target_period


def fetch_stock_historical_data(
        ticker_symbol: str,
        period: str
) -> None:
    """
    Use Yahoo Finance API to fetch stock historical data. Outputs historical data to CSV file with name SYMBOL.csv
    :param ticker_symbol: stock ticker symbols (i.e. BNP.PA for BNP Paribas)
    :param period: historical time period to load
    """
    # Fetching stock data using Yahoo Finance API
    stock_data = yf.Ticker(ticker_symbol).history(period=period)[['Open', 'High', 'Low', 'Close']]
    if not stock_data.empty:
        stock_data = stock_data.reset_index()  # Reset index of returned dataframe
        stock_data.dropna(axis=0, inplace=True)  # Drop NA value rows
        print(f'success for {ticker_symbol}')  # Log success
        stock_data.to_csv(f'../data/historical/{ticker_symbol}.csv')
    # If returned data is empty, it means that the passed ticker symbol is invalid
    else:
        print(f'Invalid ticker symbol: {ticker_symbol}')  # Log error


def fetch_market_historical_data(
        market_index_ticker_symbol: str,
        stock_ticker_symbols: list,
        period: str
) -> None:
    """
    Use Yahoo Finance API to fetch stock historical data
    :param market_index_ticker_symbol: ticker symbol for market index
    :param stock_ticker_symbols: list of target stock ticker symbols
    :param period: historical time period to load
    """
    # Iterate over historical data directory
    for item in os.listdir('../data/historical/'):
        # Remove unwanted historical data files
        if item.split('.csv')[0] not in stock_ticker_symbols:
            os.remove(os.path.join('../data/historical/', item))
    # Fetch historical data for every ticker symbol
    for symbol in stock_ticker_symbols:
        fetch_stock_historical_data(ticker_symbol=symbol, period=period)
    # Fetch market index historical data
    fetch_stock_historical_data(ticker_symbol=market_index_ticker_symbol, period=period)


if __name__ == '__main__':
    fetch_market_historical_data(
        market_index_ticker_symbol=market_symbol,
        stock_ticker_symbols=target_symbols,
        period=target_period
    )
