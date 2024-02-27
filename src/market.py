import pandas as pd
import yfinance as yf
import numpy as np
from src.parameters import target_period, target_symbols, risk_free_rate, minimum_share_price
from src.portfolio import Portfolio


class Market(object):
    """
    The Market
    """
    def __init__(self) -> None:
        """
        Market constructor
        """
        self.period: str = target_period
        self.stock_symbols: list[str] = target_symbols
        self.risk_free_rate: float = risk_free_rate
        self.minimum_share_price: float = minimum_share_price
        self.data: pd.DataFrame | None = None

    def load_data(self) -> None:
        """
        Load the data from Yahoo Finance API
        """
        self.data = yf.download(tickers=self.stock_symbols, period=self.period)['Adj Close']
        stocks_to_keep = [col for col in self.data.columns if self.data[col].iloc[0] >= self.minimum_share_price]
        self.data = self.data[stocks_to_keep]
        self.data.sort_values(by='Date', ascending=False, inplace=True)

    def extract_top_n_stocks(
            self,
            n_stocks: int
    ) -> list[str]:
        """
        Extract the top n stocks
        :param n_stocks: number of stocks to extract
        :return: list of stock ticker symbols
        """
        market_returns = self.data.pct_change(periods=-1)[:-1].fillna(method='ffill', axis=0)
        mean_returns = list(market_returns.mean())
        top_indices = np.argsort(mean_returns)[-n_stocks:][::-1]
        return self.data.columns[top_indices].tolist()


if __name__ == '__main__':
    market = Market()
    market.load_data()
    top_n_stocks = market.extract_top_n_stocks(5)
    pf = Portfolio()
    for stock in top_n_stocks:
        pf.add_stock(stock)

