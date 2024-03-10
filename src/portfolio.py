import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from scipy.optimize import minimize, Bounds, LinearConstraint

from src.parameters import capital, number_of_assets, optimisation_factor, risk_free_rate, minimum_proportion
from src.utilities import filter_portfolio_evolution
from src.stock import Stock
from src.market import Market


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
        self.bearish_stocks: list[Stock] = []
        self.stock_returns: pd.DataFrame = pd.DataFrame()
        self.expected_return: float | None = None
        self.risk: float | None = None
        self.sharpe_ratio: float | None = None
        self.value: float | None = None

    def __repr__(self) -> str:
        """
        String representation of the Portfolio object
        """
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

    def add_stocks(
            self,
            stocks: list[Stock] | list[str]
    ) -> None:
        """
        Add multiple stocks to the portfolio
        :param stocks: list of stocks to add, or ticker symbols
        """
        for stock in stocks:
            self.add_stock(stock)

    def remove_stock(
            self,
            stock: Stock
    ) -> None:
        """
        Remove a stock from the portfolio
        :param stock: Stock to remove, or ticker symbol
        :return:
        """
        if isinstance(stock, Stock):
            self.stocks.remove(stock)
        elif isinstance(stock, str):
            for s in self.stocks:
                if s.ticker == stock:
                    self.stocks.remove(s)

    def optimise(self) -> None:
        """
        Optimise the portfolio
        """
        self.stock_returns = pd.concat(objs=[s.data for s in self.stocks], axis=1)
        self.stock_returns.sort_values(by='Date', ascending=False, inplace=True)
        self.stock_returns = self.stock_returns.pct_change(periods=-1)[:-1].fillna(method='bfill', axis=0)
        # Set up the optimisation problem
        weights = 1 / self.n_assets * np.ones(self.n_assets)
        weight_bounds = Bounds(lb=minimum_proportion * np.ones(self.n_assets), ub=np.ones(self.n_assets))
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
        self.sharpe_ratio = (self.expected_return - risk_free_rate) / self.risk
        self.value = 0
        for stock in self.stocks:
            self.value += stock.price * stock.shares

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
        pf.sharpe_ratio = dictionary['characteristics']['sharpeRatio']
        pf.capital = dictionary['characteristics']['capital']
        pf.value = dictionary['characteristics']['value']
        for ticker, stock_data in dictionary['stocks'].items():
            stock = Stock.from_dict(ticker=ticker, dictionary=stock_data)
            pf.add_stock(stock)
        pf.value = 0
        return pf

    @staticmethod
    def load_current_portfolio():
        """
        Load portfolio from local data
        """
        # Loading portfolio
        with open('../data/current_portfolio.json', 'r') as f:
            return Portfolio.from_dict(json.load(f))

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
        pf_dict['characteristics']['sharpeRatio'] = self.sharpe_ratio
        pf_dict['characteristics']['capital'] = self.capital
        pf_dict['characteristics']['value'] = self.value
        for stock in self.stocks:
            pf_dict['stocks'][stock.ticker] = stock.to_dict()
        return pf_dict

    def update(self):
        """
        Update portfolio in JSON file
        """
        # Output portfolio to JSON file
        with open('../data/current_portfolio.json', 'w') as file:
            json.dump(self.to_dict(), file, indent=4)
        with open('../config/portfolio_config.json', 'r') as file:
            portfolio_config = json.load(file)
            portfolio_config['capital'] = self.value
        with open('../config/portfolio_config.json', 'w') as file:
            json.dump(portfolio_config, file, indent=4)

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
        plt.title(label='Portfolio Composition by Capital', fontweight='bold')
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

    def evaluate(self):
        """
        Evaluate the portfolio value
        """
        current_value = 0
        for stock in self.stocks:
            stock.evaluate()
            current_value += stock.price * stock.shares
        self.value = current_value

    def update_evolution(self):
        """
        Append the portfolio value to the portfolio historical values file
        """
        current_date = datetime.now().date().strftime('%Y-%m-%d')
        if os.path.exists('../data/portfolio_evolution.json'):
            with open('../data/portfolio_evolution.json', 'r') as f:
                pf_values = json.load(f)
                pf_values['date'].append(current_date)
                pf_values['value'].append(self.value)
                pf_values = filter_portfolio_evolution(pf_values=pf_values)
            with open('../data/portfolio_evolution.json', 'w') as f:
                json.dump(pf_values, f, indent=4)
        else:
            pf_values = {'date': [current_date], 'value': [self.value]}
            with open('../data/portfolio_evolution.json', 'w') as f:
                json.dump(pf_values, f, indent=4)

    @staticmethod
    def plot_evolution():
        """
        Plot the portfolio evolution
        """
        with open('../data/portfolio_evolution.json', 'r') as f:
            pf_values = json.load(f)
            plt.figure(figsize=(20, 10))
            sns.lineplot(x=pf_values['date'], y=pf_values['value'])
            plt.title('Evolution of portfolio value')
            plt.xlabel('Date')
            plt.ylabel('Value in Euros')
            plt.tight_layout()
            plt.savefig('../graphs/portfolio/portfolio_evolution.png')
            plt.close()

    def predict(self):
        """
        Predict the portfolio stocks movements
        """
        for stock in self.stocks:
            if stock.bearish():
                self.bearish_stocks.append(stock)
                self.remove_stock(stock)

    def suggest_action(self):
        """
        Predict the portfolio evolution and output a suggestion
        """
        # Predict portfolio stocks
        self.predict()
        if len(self.bearish_stocks) == 0:
            # Evaluate portfolio and update local data if portfolio is bullish
            self.evaluate()
            self.update()
            self.update_evolution()
            self.plot_evolution()
            print('Bullish portfolio, no action to suggest')
            return None
        else:
            print('Bearish stocks in portfolio:')
            # Print bearish stocks
            for bearish_stock in self.bearish_stocks:
                print(f' - {bearish_stock.ticker}')
            # Instantiate suggestion dictionary
            action = {
                'sell': dict(),
                'suggestedPortfolio': dict()
            }
            # Add bearish portfolio stocks to 'sell' dictionary
            for stock in self.bearish_stocks:
                action['sell'][stock.ticker] = stock.shares
            # Instantiate market and remove used stock tickers
            market = Market()
            market.remove_stock_symbols([s.ticker for s in self.bearish_stocks])
            market.remove_stock_symbols([s.ticker for s in self.stocks])
            # Save old stock objects
            old_stocks = self.stocks
            # Extract best predicted stocks and add them to portfolio
            top_performing_stocks = market.extract_top_n_stocks(
                n_stocks=len(self.bearish_stocks),
                choice_method='prediction'
            )
            for stock in top_performing_stocks:
                self.add_stock(stock)
            # Optimise portfolio
            self.optimise()
            # Add modifications to 'sell' or 'buy' dictionary
            for old_stock in old_stocks:
                for new_stock in self.stocks:
                    if old_stock.ticker == new_stock.ticker and new_stock.shares < old_stock.shares:
                        action['sell'][new_stock.ticker] = old_stock.shares - new_stock.shares
            # Evaluate suggested portfolio value
            self.evaluate()
            # Dump suggestion to JSON
            action['suggestedPortfolio'] = self.to_dict()
            with open('../data/suggested_portfolio.json', 'w') as f:
                json.dump(action, f, indent=4)

    @staticmethod
    def load_suggested_portfolio():
        """
        Loads suggested portfolio
        """
        # Loading portfolio
        with open('../data/suggested_portfolio.json', 'r') as f:
            return Portfolio.from_dict(json.load(f)['suggestedPortfolio'])

    def act_on_suggestion(self):
        """
        Acts on suggested portfolio
        """
        with open('../data/suggested_portfolio.json', 'r') as f:
            pf = self.from_dict(json.load(f)['suggestedPortfolio'])
        pf.update()
        pf.update_evolution()
        pf.plot_evolution()
