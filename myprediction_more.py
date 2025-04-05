import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
import requests
from datetime import datetime
from bs4 import BeautifulSoup

# Fetch stock data
stock = 'ARVIND.NS'
start_date='2020-01-01'
#end_date = '2025-04-01'  # Adjusted to a future date for testing
end_date= datetime.strftime(datetime.now(), "%Y-%m-%d")

data = yf.download(stock, start=start_date, end=end_date)
if data.empty:
    raise ValueError(f"No data fetched for {stock}.")
print(f"Initial data shape: {data.shape}")

# Fetch NIFTY 50 data
nifty = yf.download('^NSEI', start=start_date, end=end_date)  # NIFTY 50 index
data['NIFTY_Close'] = nifty['Close'].reindex(data.index, method='ffill')  # Align with stock dates

# Fetch INR/USD exchange rate (using USDINR=X ticker)
inr_usd = yf.download('USDINR=X', start=start_date, end=end_date)
data['INR_USD'] = inr_usd['Close'].reindex(data.index, method='ffill')

# Select relevant columns
data = data[['Close', 'Volume', 'NIFTY_Close', 'INR_USD']]

# Technical indicators
data['SMA_20'] = data['Close'].rolling(window=20).mean()
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
    return rsi
data['RSI'] = calculate_rsi(data['Close'])

# MACD
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Fundamental data
stock_info = yf.Ticker(stock)
data['PE_Ratio'] = stock_info.info.get('trailingPE', 0)
data['EPS'] = stock_info.info.get('trailingEps', 0)  # Earnings Per Share

# Sentiment analysis
def get_sentiment(stock):
    try:
        url = f'https://news.google.com/search?q={stock}+stock'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3')[:5]]
        if not headlines:
            return 0
        sentiment = np.mean([TextBlob(headline).sentiment.polarity for headline in headlines])
        return sentiment
    except Exception as e:
        print(f"Sentiment error: {e}")
        return 0
data['Sentiment'] = get_sentiment(stock)

# Prepare data
data = data.dropna()
print(f"Data shape after dropping NaN: {data.shape}")
if data.empty or len(data) < 27:  # Adjust for EMA_26
    raise ValueError("Not enough data after dropping NaN. Extend date range.")

# Features and target
features = ['SMA_20', 'RSI', 'Volume', 'PE_Ratio', 'Sentiment', 'NIFTY_Close', 'INR_USD', 'MACD', 'Signal_Line', 'EPS']
X = data[features].values
y = data['Close'].shift(-1).dropna().values.ravel()
X = X[:-1]
print(f"X shape: {X.shape}, y shape: {y.shape}")

if len(X) == 0 or len(y) == 0:
    raise ValueError("X or y is empty.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
latest_data = X[-1].reshape(1, -1)
predicted_price = model.predict(latest_data)[0]
print(f'Predicted next day price for {stock}: ₹{predicted_price:.2f}')

# Recommendation
current_price = data['Close'].iloc[-1].item()
sma_20 = data['SMA_20'].iloc[-1].item()
rsi = data['RSI'].iloc[-1].item()
sentiment = data['Sentiment'].iloc[-1].item()
macd = data['MACD'].iloc[-1].item()
signal_line = data['Signal_Line'].iloc[-1].item()

recommendation = 'Hold'
if (predicted_price > current_price * 1.05 and rsi < 70 and sentiment > 0 and macd > signal_line):
    recommendation = 'Buy'
elif (predicted_price < current_price * 0.95 or rsi > 70 or macd < signal_line):
    recommendation = 'Sell'
print(f'Recommendation for {stock}: {recommendation}')