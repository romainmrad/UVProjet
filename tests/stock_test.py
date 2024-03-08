import unittest
import numpy as np
import pandas as pd

from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from src.stock import Stock


class TestStock(unittest.TestCase):
    def setUp(self):
        # Initialize test stocks
        self.test_ticker = "MC.PA"
        self.test_stock = Stock(self.test_ticker)

    def test_init(self):
        # Test the __init__ method
        self.assertEqual(self.test_stock.ticker, "MC.PA")
        self.assertIsInstance(self.test_stock.data, pd.Series)
        self.assertIsInstance(self.test_stock.price, float)
        self.assertIsInstance(self.test_stock.expected_return, float)
        self.assertIsInstance(self.test_stock.risk, float)
        self.assertIsInstance(self.test_stock.sharpe_ratio, float)
        self.assertIsNone(self.test_stock.weight)
        self.assertIsNone(self.test_stock.capital)
        self.assertIsNone(self.test_stock.shares)

    def test_repr(self):
        # Test the __repr__ method
        stock_repr = repr(self.test_stock)
        self.assertIsInstance(stock_repr, str)
        self.assertIn(self.test_ticker, stock_repr)

    def test_evaluate(self):
        # Test the evaluate method
        self.test_stock.evaluate()
        self.assertIsInstance(self.test_stock.price, float)

    def test_set_weight(self):
        # Test the set_weight method
        weight = 0.5
        self.test_stock.set_weight(weight)
        self.assertEqual(self.test_stock.weight, weight)
        self.assertIsNotNone(self.test_stock.capital)
        self.assertIsNotNone(self.test_stock.shares)

    def test_get_company_name(self):
        # Test the get_company_name method
        company_name = self.test_stock.get_company_name()
        self.assertIsInstance(company_name, str)

    def test_from_dict(self):
        # Test the from_dict method
        stock_dict = self.test_stock.to_dict()
        new_stock = Stock.from_dict(self.test_ticker, stock_dict)
        self.assertEqual(stock_dict, new_stock)

    def test_to_dict(self):
        # Test the to_dict method
        stock_dict = self.test_stock.to_dict()
        self.assertIsInstance(stock_dict, dict)
        self.assertIn('companyName', stock_dict)
        self.assertIn('pricePerShare', stock_dict)
        self.assertIn('return', stock_dict)
        self.assertIn('risk', stock_dict)
        self.assertIn('sharpeRatio', stock_dict)
        self.assertIn('capital', stock_dict)
        self.assertIn('weight', stock_dict)
        self.assertIn('shares', stock_dict)

    def test_get_train_data(self):
        # Test the get_train_data method
        train_data = Stock.get_train_data(self.test_ticker)
        self.assertIsInstance(train_data, np.ndarray)

    def test_get_prediction_data(self):
        # Test the get_prediction_data method
        prediction_data = self.test_stock.get_prediction_data()
        self.assertIsInstance(prediction_data, np.ndarray)

    def test_load_model_and_scaler(self):
        # Test the load_model_and_scaler method
        model, scaler = self.test_stock.load_model_and_scaler()
        self.assertIsInstance(model, Sequential)
        self.assertIsInstance(scaler, MinMaxScaler)

    def test_predict(self):
        # Test the predict method
        prediction = self.test_stock.predict()
        self.assertIsInstance(prediction, (float, type(None)))


if __name__ == '__main__':
    unittest.main()
