from src.market import Market


if __name__ == '__main__':
    # Initialise Market Object
    market = Market()
    # Select best model for each stock and save it to file
    market.train_models()
