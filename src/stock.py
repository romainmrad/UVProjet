import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.parameters import risk_free_rate, capital
from src.utilities import train_test_split


class Stock(object):
    """
    Stock class
    """
    def __init__(
            self,
            ticker: str,
            load_data: bool = True,
    ) -> None:
        """
        Stock constructor
        :param ticker: stock ticker symbol
        :param load_data: whether to load the data or not
        """
        self.ticker: str = ticker
        self.data: pd.DataFrame or None = None
        self.price: float or None = None
        self.expected_return: float or None = None
        self.risk: float or None = None
        self.sharpe_ratio: float or None = None
        self.weight: float or None = None
        self.capital: float or None = None
        if load_data:
            self.data = yf.download(self.ticker, period='1mo')['Adj Close']
            self.price = self.data.iloc[-1]
            self.expected_return = self.data.pct_change(periods=1).dropna().mean()
            self.risk = self.data.pct_change(periods=1).dropna().var()
            self.sharpe_ratio = (self.expected_return - risk_free_rate) / self.risk

    def set_weight(self, weight) -> None:
        """
        Set stock weight and capital
        :param weight: the weight of the stock
        """
        self.weight = weight
        self.capital = self.weight * capital

    def get_company_name(self) -> str | None:
        try:
            # Create a Ticker object for the given symbol
            ticker = yf.Ticker(self.ticker)
            # Get the info dictionary for the ticker
            info = ticker.info
            # Extract the company name
            if 'shortName' in info.keys():
                return info['shortName']
            elif 'longName' in info.keys():
                return info['longName']
            return None
        except Exception as e:
            print(f"Error: {e} for {self.ticker}")
            return None

    @staticmethod
    def from_dict(
            ticker: str,
            dictionary: dict
    ):
        """
        Create stock object
        :param ticker: ticker symbol
        :param dictionary: data
        :return: stock object
        """
        s = Stock(ticker=ticker)
        s.price = dictionary['pricePerShare']
        s.expected_return = dictionary['return']
        s.risk = dictionary['risk']
        s.sharpe_ratio = dictionary['sharpeRatio']
        s.capital = dictionary['capital']
        s.weight = dictionary['weight']
        return s

    def to_dict(self) -> dict:
        """
        Converts the stock object to a dictionary
        :return: Stock data as dictionary
        """
        stock_dict = dict()
        stock_dict['companyName'] = self.get_company_name()
        stock_dict['pricePerShare'] = self.price
        stock_dict['return'] = self.expected_return
        stock_dict['risk'] = self.risk
        stock_dict['sharpeRatio'] = self.sharpe_ratio
        stock_dict['capital'] = self.capital
        stock_dict['weight'] = self.weight
        stock_dict['shares'] = int(stock_dict['capital'] // stock_dict['pricePerShare'])
        return stock_dict

    def predict(self):
        stock_data = yf.download(tickers=self.ticker, period='5y')['Adj Close']
        stock_data.columns = [self.ticker]
        stock_data.dropna(inplace=True)
        scaler = MinMaxScaler()
        scaled_stock_data = scaler.fit_transform(np.array(stock_data).reshape(-1, 1))
        x_train, x_test, y_train, y_test = train_test_split(dataset=scaled_stock_data, time_step=100)
