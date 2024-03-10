from src.portfolio import Portfolio


if __name__ == "__main__":
    # Load current portfolio
    pf = Portfolio.load_suggested_portfolio()
    # Evaluate portfolio value
    pf.evaluate()
    # Update current portfolio
    pf.update()
    pf.plot()
    print('Updated portfolio according to suggestion')
    pf.update_evolution()
    print('Updated portfolio evolution JSON')
    pf.plot_evolution()
    print('Plotted portfolio evolution JSON')
