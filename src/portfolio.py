import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, Bounds, LinearConstraint
from src.parameters import capital, number_of_assets, optimisation_factor, risk_free_rate
from src.stock import Stock


class Portfolio(object):
    """
    The Portfolio
    """
    def __init__(self) -> None:
        """
        Portfolio constructor
        """
        self.capital: float = capital
        self.n_assets: int = number_of_assets
        self.optimisation_factor: str = optimisation_factor
        self.weights: list[float] = []
        self.stocks: list[Stock] = []
        self.stock_returns: pd.DataFrame = pd.DataFrame()
        self.expected_return: float | None = None
        self.risk: float | None = None
        self.share_ratio: float | None = None

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def add_stock(
            self,
            stock: Stock | str
    ) -> None:
        """
        Add a stock to the portfolio
        :param stock: Stock to add, or ticker symbol
        """
        if isinstance(stock, Stock):
            self.stocks.append(stock)
        elif isinstance(stock, str):
            stock = Stock(stock)
            self.stocks.append(stock)

    def optimise(self) -> None:
        """
        Optimise the portfolio
        """
        self.stock_returns = pd.concat(objs=[s.data for s in self.stocks], axis=1)
        self.stock_returns.sort_values(by='Date', ascending=False, inplace=True)
        self.stock_returns = self.stock_returns.pct_change(periods=-1)[:-1].fillna(method='bfill', axis=0)
        # Set up the optimisation problem
        weights = 1 / self.n_assets * np.ones(self.n_assets)
        weight_bounds = Bounds(lb=0.1 * np.ones(self.n_assets), ub=np.ones(self.n_assets))
        linear_constraint = LinearConstraint(np.ones(self.n_assets), lb=1, ub=1)
        # Solve optimisation problem
        optimisation = minimize(
            fun=self.compute_optimisation_factor,
            x0=weights,
            method='trust-constr',
            constraints=linear_constraint,
            args=(self.stock_returns, optimisation_factor, risk_free_rate),
            bounds=weight_bounds
        )
        self.weights = optimisation.x
        for i, stock in enumerate(self.stocks):
            stock.set_weight(self.weights[i])

    def compute_characteristics(self) -> None:
        """
        Compute portfolio characteristics
        """
        weighted_returns = (self.stock_returns * self.weights).sum(axis=1)
        self.expected_return = weighted_returns.mean()
        self.risk = weighted_returns.var()
        self.share_ratio = (self.expected_return - risk_free_rate) / self.risk

    @staticmethod
    def from_dict(
            dictionary: dict
    ):
        """
        Create portfolio
        :param dictionary: data
        :return: portfolio object
        """
        pf = Portfolio()
        pf.n_assets = dictionary['characteristics']['numberOfAssets']
        pf.expected_return = dictionary['characteristics']['return']
        pf.risk = dictionary['characteristics']['risk']
        pf.share_ratio = dictionary['characteristics']['sharpeRatio']
        pf.capital = dictionary['characteristics']['capital']
        for ticker, stock_data in dictionary['stocks'].items():
            stock = Stock(ticker=ticker, load_data=False).from_dict(ticker=ticker, dictionary=stock_data)
            pf.add_stock(stock)
        return pf

    def to_dict(self) -> dict:
        """
        Convert portfolio to dictionary
        :return: Portfolio data as dictionary
        """
        pf_dict = dict()
        pf_dict['characteristics'] = dict()
        pf_dict['stocks'] = dict()
        pf_dict['characteristics']['numberOfAssets'] = self.n_assets
        pf_dict['characteristics']['return'] = self.expected_return
        pf_dict['characteristics']['risk'] = self.risk
        pf_dict['characteristics']['sharpeRatio'] = self.share_ratio
        pf_dict['characteristics']['capital'] = self.capital
        for stock in self.stocks:
            pf_dict['stocks'][stock.ticker] = stock.to_dict()
        return pf_dict

    def plot(self) -> None:
        """
        Plot portfolio
        """
        portfolio_stock_data = {
            'company': [],
            'investedCapital': []
        }
        for stock in self.stocks:
            portfolio_stock_data['company'].append(stock.get_company_name())
            portfolio_stock_data['investedCapital'].append(stock.weight * self.capital)
        df = pd.DataFrame(portfolio_stock_data)
        # Plotting
        plt.figure(figsize=(10, 10))
        colors = sns.color_palette('YlGnBu')[0:len(df)]
        plt.pie(
            df['investedCapital'],
            labels=df['company'],
            autopct=lambda x: '{:.0f}€'.format(x * df['investedCapital'].sum() / 100),
            colors=colors,
            startangle=90
        )
        plt.title('Portfolio Composition by Capital', fontweight='bold')
        # Check if the path exists
        if not os.path.exists('../graphs/portfolio/'):
            # Create the directory
            os.makedirs('../graphs/portfolio/')
        # Save the figure to a file path
        plt.savefig('../graphs/portfolio/current_portfolio.png')
        plt.close()

    @staticmethod
    def compute_optimisation_factor(
            x: np.ndarray,
            stock_returns: pd.DataFrame,
            factor: str,
            rfr: float
    ) -> float:
        """
        Compute negative Sharpe ratio for a portfolio of weights x
        :param x: the weights of the portfolio
        :param stock_returns: dataframe of stock daily returns
        :param factor: the factor to optimise. Values are 'SharpeRatio', 'Risk' and 'Return'
        :param rfr: risk-free rate for ratio computing
        :return: Sharpe Ratio, Risk or Exected Return
        """
        weighted_returns = (stock_returns * x).sum(axis=1)
        match factor:
            case 'sharpeRatio':
                expected_return = weighted_returns.mean()
                risk = weighted_returns.var()
                return - (expected_return - rfr) / risk
            case 'risk':
                return weighted_returns.var()
            case 'return':
                return - weighted_returns.mean()
            case _:
                raise ValueError(f'{factor} is not a valid optimisation factor')
