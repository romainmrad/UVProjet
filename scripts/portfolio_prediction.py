import json

from src.portfolio import Portfolio


if __name__ == "__main__":
    # Loading current portfolio
    pf = Portfolio.load_current_portfolio()
    # Predicting stock movements
    pf.suggest_action()
