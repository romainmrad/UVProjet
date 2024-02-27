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
    pf_data = pf.to_dict()
    # Output portfolio to JSON file
    with open('../data/portfolio/current_portfolio.json', 'w') as file:
        json.dump(pf_data, file, indent=4)
    print('Portfolio JSON file created')
    # Plotting portfolio
    pf.plot()
    print('Portfolio plot image created')
