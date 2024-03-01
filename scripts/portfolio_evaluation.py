import json
from src.portfolio import Portfolio


if __name__ == '__main__':
    # Loading portfolio
    with open('../data/portfolio/current_portfolio.json', 'r') as f:
        pf = Portfolio.from_dict(json.load(f))
    pf.evaluate()
    pf.update()
    pf.update_evolution()
    pf.plot_evolution()
