import joblib
import pandas as pd
import yfinance as yf
import numpy as np

from keras.backend import clear_session
from sklearn.preprocessing import MinMaxScaler

from src.utilities import train_test_split, model_selection, plot_prediction
from src.parameters import target_period, target_symbols, risk_free_rate, minimum_share_price, lookback, prediction_metric
from src.stock import Stock


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

    def remove_stock_symbols(
            self,
            symbols: str | list[str] = None
    ) -> None:
        """
        Remove stock.s from market
        :param symbols: symbol.s to remove
        """
        # If passed one symbol
        if isinstance(symbols, str):
            self.stock_symbols.remove(symbols)
        # If passed a list of symbols
        elif isinstance(symbols, list):
            for symbol in symbols:
                self.stock_symbols.remove(symbol)

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

    def extract_top_n_predicted_stocks(
            self,
            n_stocks: int
    ) -> list[str]:
        predictions = []
        predicted_stock_ticker_symbols = []
        for ticker in self.stock_symbols:
            current_stock = Stock(ticker)
            current_prediction = current_stock.predict()
            if current_prediction is not None and current_prediction > current_stock.price:
                predictions.append(current_prediction - current_stock.price)
                predicted_stock_ticker_symbols.append(ticker)
        top_indices = np.argsort(predictions)[-n_stocks:][::-1]
        return [predicted_stock_ticker_symbols[i] for i in top_indices]

    def train_models(self):
        # Iterating over all market stocks
        for ticker in self.stock_symbols:
            # Fetching stock historical data
            stock_data = Stock.get_train_data(ticker=ticker)
            # Instantiating scaler and scaling data
            scaler = MinMaxScaler()
            scaled_stock_data = scaler.fit_transform(stock_data)
            # Split data for training and testing
            x_train, x_test, y_train, y_test = train_test_split(dataset=scaled_stock_data, lookback=lookback)
            if len(y_train) >= 300:
                model, y_pred, score = model_selection(
                    ticker=ticker,
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    metric=prediction_metric
                )
                if model is not None:
                    plot_prediction(
                        ticker=ticker,
                        model=model,
                        y_true=scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                        y_pred=scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten(),
                        score=score
                    )
                    model.save(f'../models/sequential/{ticker}.keras')
                    joblib.dump(value=scaler, filename=f'../models/scaler/{ticker}.save')
                    print(f'Success for {ticker}\n')
                else:
                    print(f'Failure to fit model for {ticker}')
                clear_session()


if __name__ == '__main__':
    market = Market()
    market.train_models()
