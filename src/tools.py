import yfinance as yf


def get_company_name(ticker_symbol):
    try:
        # Create a Ticker object for the given symbol
        ticker = yf.Ticker(ticker_symbol)
        # Get the info dictionary for the ticker
        info = ticker.info
        # Extract the company name
        company_name = info['longName']
        return company_name
    except Exception as e:
        print(f"Error: {e}")
        return None
