import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.tools import get_company_name


def plot_portfolio_visualisation(portfolio_data: dict) -> None:
    """
    Visualise portfolio diversification using invested amount and number of shares
    :param portfolio_data: portfolio data
    """
    portfolio_stock_data = {
        'company': [],
        'investedCapital': [],
        'shares': []
    }
    for stock, stock_data in portfolio_data['stocks'].items():
        portfolio_stock_data['company'].append(get_company_name(stock))
        portfolio_stock_data['investedCapital'].append(round(stock_data['investedCapital'], 2))
        portfolio_stock_data['shares'].append(stock_data['shares'])

    df = pd.DataFrame(portfolio_stock_data)
    # Plotting
    plt.figure(figsize=(20, 10))
    colors = sns.color_palette('YlGnBu')[0:len(df)]

    # Pie chart for the Amount column
    plt.subplot(1, 2, 1)
    plt.pie(
        df['investedCapital'],
        labels=df['company'],
        autopct=lambda x: '{:.0f}€'.format(x*df['investedCapital'].sum()/100),
        colors=colors,
        startangle=90
    )
    plt.title('Portfolio Composition by Capital', fontweight='bold')

    # Pie chart for the Shares column
    plt.subplot(1, 2, 2)
    plt.pie(
        df['shares'],
        labels=df['company'],
        autopct=lambda x: '{:.0f}'.format(x * df['shares'].sum() / 100),
        colors=colors,
        startangle=90
    )
    plt.title('Portfolio Composition by Shares', fontweight='bold')

    # Check if the path exists
    if not os.path.exists('../graphs/portfolio/'):
        # Create the directory
        os.makedirs('../graphs/portfolio/')
    # Save the figure to a file path
    plt.savefig('../graphs/portfolio/current_portfolio.png')
    plt.close()
