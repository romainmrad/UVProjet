import yfinance as yf

data = yf.download(ticker='MLCOT.PA', period='1D')['Adj Close'].values[0]
print(data)
