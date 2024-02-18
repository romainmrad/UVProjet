import yfinance as yf


def get_company_name(ticker_symbol):
    try:
        # Create a Ticker object for the given symbol
        ticker = yf.Ticker(ticker_symbol)
        # Get the info dictionary for the ticker
        info = ticker.info
        # Extract the company name
        if 'shortName' in info.keys():
            return info['shortName']
        elif 'longName' in info.keys():
            return info['longName']
        return None
    except Exception as e:
        print(f"Error: {e} for {ticker_symbol}")
        return None
