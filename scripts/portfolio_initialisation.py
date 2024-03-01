import json
from src.parameters import number_of_assets
from src.market import Market
from src.portfolio import Portfolio


if __name__ == '__main__':
    # Initialising market
    market = Market()
    market.load_data()
    # Extracting top n stocks
    top_n_stocks = market.extract_top_n_stocks(n_stocks=number_of_assets)
    # Initialising portfolio
    pf = Portfolio()
    # Adding stocks to portfolio
    for stock in top_n_stocks:
        pf.add_stock(stock=stock)
    # Optimising portfolio weights
    pf.optimise()
    # Computing portfolio characteristics
    pf.compute_characteristics()
    # Formatting portfolio data as JSON
    pf.update()
    print('Portfolio JSON file created')
    # Plotting portfolio
    pf.plot()
    print('Portfolio plot image created')
