from src.portfolio import Portfolio


if __name__ == '__main__':
    # Load current portfolio
    pf = Portfolio.load_current_portfolio()
    # Evaluate portfolio value and update local data and plot
    pf.evaluate()
    pf.update()
    pf.plot()
    print('Portfolio evaluation completed')
    pf.update_evolution()
    print('Updated portfolio evolution JSON')
    pf.plot_evolution()
    print('Plotted portfolio evolution JSON')
