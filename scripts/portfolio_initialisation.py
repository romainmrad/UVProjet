from src.parameters import number_of_assets, initialisation_method
from src.market import Market
from src.portfolio import Portfolio


if __name__ == '__main__':
    # Initialising market
    market = Market()
    market.load_data()
    # Extracting top n stocks
    top_n_stocks = market.extract_top_n_stocks(n_stocks=number_of_assets, choice_method=initialisation_method)
    # Initialising portfolio
    pf = Portfolio()
    # Adding stocks to portfolio
    pf.add_stocks(stocks=top_n_stocks)
    # Optimising portfolio weights
    pf.optimise()
    # Computing portfolio characteristics
    pf.compute_characteristics()
    # Formatting portfolio data as JSON
    pf.evaluate()
    pf.update()
    pf.plot()
    print('Portfolio evaluation completed')
    pf.update_evolution()
    print('Updated portfolio evolution JSON')
    pf.plot_evolution()
    print('Plotted portfolio evolution JSON')
