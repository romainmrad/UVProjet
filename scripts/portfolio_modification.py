from src.portfolio import Portfolio


if __name__ == "__main__":
    # Load current portfolio
    pf = Portfolio.load_suggested_portfolio()
    # Evaluate portfolio value
    pf.evaluate()
    # Update current portfolio
    pf.update()
