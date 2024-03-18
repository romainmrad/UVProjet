import json
import joblib
import pandas as pd
import yfinance as yf
import numpy as np

from keras.models import load_model, Sequential
from sklearn.preprocessing import MinMaxScaler

from src.parameters import risk_free_rate, capital, trading_fee, lookback


class Stock(object):
    """
    Stock class
    """

    def __init__(
            self,
            ticker: str,
    ) -> None:
        """
        Stock constructor
        :param ticker: stock ticker symbol
        """
        self.ticker: str = ticker
        self.data: pd.DataFrame = yf.download(self.ticker, period='1mo')['Adj Close']
        self.price: float = self.data.iloc[-1]
        self.expected_return: float = self.data.pct_change(periods=1).dropna().mean()
        self.risk: float = self.data.pct_change(periods=1).dropna().var()
        self.sharpe_ratio: float = (self.expected_return - risk_free_rate) / self.risk
        self.weight: float | None = None
        self.capital: float | None = None
        self.shares: int | None = None

    def __repr__(self) -> str:
        """
        String representation of the Stock object
        """
        stock_dict = {self.ticker: self.to_dict()}
        return json.dumps(stock_dict, indent=4)

    def evaluate(self):
        """
        Evaluates the stock value
        """
        self.price = yf.download(self.ticker, period='1mo')['Adj Close'].iloc[-1]

    def set_weight(self, weight) -> None:
        """
        Set stock weight and capital
        :param weight: the weight of the stock
        """
        self.weight = weight
        self.capital = self.weight * capital
        self.shares = int(self.capital // self.price)

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
        s.shares = dictionary['shares']
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
        stock_dict['capital'] = self.capital if not None else None
        stock_dict['weight'] = self.weight if not None else None
        if stock_dict['capital'] is not None and stock_dict['weight'] is not None:
            stock_dict['shares'] = int(stock_dict['capital'] // stock_dict['pricePerShare'])
        else:
            stock_dict['shares'] = None
        return stock_dict

    @staticmethod
    def get_train_data(ticker: str) -> np.ndarray:
        stock_data = yf.download(tickers=ticker, period='5y')['Adj Close']
        stock_data.dropna(inplace=True)
        return stock_data.to_numpy().reshape(-1, 1)

    def get_prediction_data(self) -> np.ndarray:
        # Fetch data using Yahoo Finance API
        stock_data = yf.download(self.ticker, period='1y')['Adj Close']
        # Drop NA values
        stock_data.dropna(inplace=True)
        # Convert to numpy
        stock_data = np.array(stock_data)[-lookback:]
        # Return reshaped array
        return stock_data

    def load_model_and_scaler(self) -> tuple[Sequential, MinMaxScaler]:
        """
        Load model and scaler for current stock
        :return: model and scaler
        """
        model = None
        scaler = None
        # Load sequential model
        try:
            model = load_model(f'../models/sequential/{self.ticker}.keras')
            print(f'Model found and loaded for {self.ticker}')
        except FileNotFoundError:
            print(f'No model trained for {self.ticker}')
        except OSError:
            print(f'No model trained for {self.ticker}')
        # Load scaler
        try:
            scaler = joblib.load(f'../models/scaler/{self.ticker}.save')
            print(f'Scaler found and loaded for {self.ticker}')
        except FileNotFoundError:
            print(f'No scaler fitted for {self.ticker}')
        except OSError:
            print(f'No scaler fitted for {self.ticker}')
        return model, scaler

    def predict(self) -> float | None:
        """
        Predict stock price
        :return: predicted stock price
        """
        # Load model and scaler from file
        model, scaler = self.load_model_and_scaler()
        # If model and scaler are found
        if model is not None and scaler is not None:
            # Load prediction X data
            stock_data = self.get_prediction_data()
            # Predict price
            prediction = model.predict(scaler.transform(stock_data.reshape(-1, 1)).reshape(1, -1), verbose=False)
            return scaler.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()[0]
        return None

    def bearish(self):
        """
        Predict stock movement
        :return: True if stock is bearish, False if bullish
        """
        return self.predict() + trading_fee > self.price

