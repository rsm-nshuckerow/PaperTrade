#!/usr/bin/env python
# coding: utf-8

# # Project Description
# 
# A simple robo-advisor that incorporates sentiment analysis from news headlines with stock positions to make recommendations and execute trades. 
# 
# ---

# ## Imports

# In[23]:


import os
import time
import requests
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from warnings import filterwarnings

import alpaca_trade_api as tradeapi
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

filterwarnings("ignore")


# ## Environment Setup

# In[24]:


load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_URL")

alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_URL, api_version='v2')


# ## Top 50 Stocks from S&P 500

# In[25]:


top_50_sp500_stocks = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'AMZN',  # Amazon.com Inc.
    'NVDA',  # NVIDIA Corporation
    'GOOGL', # Alphabet Inc. (Class A)
    'GOOG',  # Alphabet Inc. (Class C)
    'TSLA',  # Tesla Inc.
    'META',  # Meta Platforms Inc.
    'BRK.B', # Berkshire Hathaway Inc. (Class B)
    'UNH',   # UnitedHealth Group Incorporated
    'JNJ',   # Johnson & Johnson
    'XOM',   # Exxon Mobil Corporation
    'V',     # Visa Inc.
    'PG',    # Procter & Gamble Co.
    'JPM',   # JPMorgan Chase & Co.
    'LLY',   # Eli Lilly and Company
    'MA',    # Mastercard Incorporated
    'HD',    # The Home Depot Inc.
    'CVX',   # Chevron Corporation
    'MRK',   # Merck & Co. Inc.
    'PEP',   # PepsiCo Inc.
    'ABBV',  # AbbVie Inc.
    'KO',    # The Coca-Cola Company
    'PFE',   # Pfizer Inc.
    'AVGO',  # Broadcom Inc.
    'COST',  # Costco Wholesale Corporation
    'MCD',   # McDonald's Corporation
    'TMO',   # Thermo Fisher Scientific Inc.
    'WMT',   # Walmart Inc.
    'DHR',   # Danaher Corporation
    'NKE',   # NIKE Inc.
    'DIS',   # The Walt Disney Company
    'ADBE',  # Adobe Inc.
    'NFLX',  # Netflix Inc.
    'VZ',    # Verizon Communications Inc.
    'CSCO',  # Cisco Systems Inc.
    'ABT',   # Abbott Laboratories
    'ACN',   # Accenture plc
    'NEE',   # NextEra Energy Inc.
    'LIN',   # Linde plc
    'TXN',   # Texas Instruments Incorporated
    'MDT',   # Medtronic plc
    'PM',    # Philip Morris International Inc.
    'WFC',   # Wells Fargo & Company
    'HON',   # Honeywell International Inc.
    'QCOM',  # QUALCOMM Incorporated
    'BMY',   # Bristol-Myers Squibb Company
    'LOW',   # Lowe's Companies Inc.
    'UNP',   # Union Pacific Corporation
    'RTX'    # Raytheon Technologies Corporation
]


# ## Functions 

# In[26]:


def get_historical_stock_data(symbols, start_date, end_date):
    # Use the correct TimeFrame object for daily data
    timeframe = tradeapi.TimeFrame.Day
    
    all_data = []
    
    for symbol in symbols:
        bars = alpaca.get_bars(
            symbol,
            timeframe=timeframe,
            start=start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            end=end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            adjustment='raw',
            feed='iex'
        )
        
        data = []
        for bar in bars:
            data.append({
                'symbol': symbol,  # Add the symbol to the data
                'time': bar.t,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v
            })
        
        all_data.extend(data)
    
    return pd.DataFrame(all_data)


# In[27]:


def get_news(symbols, days_prior=1):
    today = datetime.utcnow()
    yesterday = today - timedelta(days=days_prior)  # Adjusted to retrieve news from the previous day
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    news = []
    page_token = None
    
    while True:
        params = {
            "symbols": ",".join(symbols),
            "start": start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "end": today.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "limit": 50,  # Maximum allowed by the API
            "page_token": page_token  # For pagination
        }
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()
            
            news.extend(result.get('news', []))
            
            # Get the next page token if available
            page_token = result.get('next_page_token')
            
            if not page_token:  # No more pages to fetch
                break
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print("Rate limit reached. Sleeping for 10 seconds...")
                time.sleep(1)  # Wait for 60 seconds before retrying
            else:
                print(f"HTTP error occurred: {e}")
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        
    return news


# In[28]:


def split_into_chunks(text, max_length=500):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)['input_ids'][0]
    # Ensure each chunk is no more than 512 tokens
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        # Make sure the chunk is exactly 512 tokens or less
        if len(chunk) > max_length:
            chunk = chunk[:max_length]
        chunks.append(chunk)
    return chunks


# In[29]:


def analyze_sentiment(text):
    chunks = split_into_chunks(text)
    sentiments = []
    for chunk in chunks:
        # Convert tokens back to text before sentiment analysis
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        sentiments.append(nlp(chunk_text)[0])
    
    # Aggregate sentiment scores (e.g., by averaging)
    avg_sentiment_score = sum(s['score'] for s in sentiments) / len(sentiments)
    # Determine overall sentiment by majority vote or averaging
    positive_scores = sum(s['score'] for s in sentiments if s['label'] == 'positive')
    negative_scores = sum(s['score'] for s in sentiments if s['label'] == 'negative')
    sentiment_label = 'positive' if positive_scores >= negative_scores else 'negative'
    
    return sentiment_label, avg_sentiment_score


# In[30]:


def load_last_update_date(file_path="stock_data.csv"):
    try:
        df = pd.read_csv(file_path)

        # Ensure that the 'date' column is converted to datetime, coercing errors
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows where 'date' couldn't be converted to datetime (i.e., NaT)
        df = df.dropna(subset=['date'])

        # Find the most recent date
        most_recent_date = df['date'].max()

        return most_recent_date
    except (FileNotFoundError, IndexError, KeyError):
        return None


# In[31]:


def save_last_update_date(date, file_path="stock_data.csv"):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.loc[df['date'].idxmax(), 'date'] = date
    df.to_csv(file_path, index=False)


# In[32]:


def get_new_news(symbols, start_date, end_date):
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    news = []
    page_token = None
    
    while True:
        params = {
            "symbols": ",".join(symbols),
            "start": start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "end": end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "limit": 50,
            "page_token": page_token
        }
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()
            
            news.extend(result.get('news', []))
            
            page_token = result.get('next_page_token')
            if not page_token:
                break
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print("Rate limit reached. Sleeping for 10 seconds...")
                time.sleep(10)
            else:
                print(f"HTTP error occurred: {e}")
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        
    return news


# In[33]:


def update_stock_data(symbols, file_path="stock_data.csv"):
    # Load the last update date
    last_update = load_last_update_date(file_path)
    if last_update is None:
        last_update = datetime.utcnow() - timedelta(days=30)  # Default to last 30 days
    else:
        last_update = pd.to_datetime(last_update)  # Convert to datetime
    
    # Get the current date
    today = datetime.utcnow()

    # Stop execution if last_update and today are the same
    if last_update.date() == today.date():
        print("Data is already up-to-date. No new data to fetch.")
        return
    
    # Retrieve news from Alpaca
    new_news = get_new_news(symbols, last_update, today)
    
    # Convert new news data to DataFrame
    new_news_df = pd.DataFrame(new_news)

    # Keep only the necessary columns and prepare the data
    if not new_news_df.empty:
        new_news_df['created_at'] = pd.to_datetime(new_news_df['created_at'])
        new_news_df['date'] = new_news_df['created_at'].dt.date
        new_news_df = new_news_df.explode('symbols')
        new_news_df['sentiment'], new_news_df['sentiment_score'] = zip(*new_news_df['headline'].apply(analyze_sentiment))
        new_news_df = new_news_df[['date', 'symbols', 'sentiment_score']]
        daily_sentiment = new_news_df.groupby(['date', 'symbols'])['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['date', 'symbol', 'average_sentiment_score']
    else:
        daily_sentiment = pd.DataFrame()

    # Fetch historical stock data
    historical_data = get_historical_stock_data(symbols, last_update, today)

    historical_data['date'] = pd.to_datetime(historical_data['time']).dt.date

    # Merge historical stock data with sentiment data
    stock_data_merged = pd.merge(historical_data, daily_sentiment, how='left', on=['date', 'symbol'])

    # Load existing stock data
    try:
        existing_data = pd.read_csv(file_path)
    except FileNotFoundError:
        existing_data = pd.DataFrame()

    # Combine the new merged data with the existing data
    combined_data = pd.concat([existing_data, stock_data_merged], ignore_index=True)
    
    # Save the updated DataFrame
    combined_data.to_csv(file_path, index=False)
    
    # Save the most recent update date
    save_last_update_date(today, file_path)


# In[34]:


def add_features(df):
    """
    Add moving averages, EMA, and time trend to the dataframe.
    """
    df = df.copy()  # Avoid SettingWithCopyWarning
    df.loc[:, '5_day_sma'] = df['close'].rolling(window=5).mean()
    df.loc[:, '5_day_ema'] = df['close'].ewm(span=5, adjust=False).mean()
    df.loc[:, 'time_trend'] = np.arange(len(df))
    return df


# In[35]:


def prepare_data(file_path='stock_data.csv'):
    """
    Prepare the data by handling missing values, adding features, and splitting into features and target.
    """
    # Load the stock data
    stock_data = pd.read_csv(file_path)

    # Convert 'time' column to datetime
    stock_data['time'] = pd.to_datetime(stock_data['time'])

    # Add new features
    stock_data = add_features(stock_data)

    # Prepare input features (X) and labels (y)
    feature_columns = ['close', 'log_sentiment_score', '5_day_sma', '5_day_ema', 'time_trend']
    X = stock_data[feature_columns]
    y = stock_data['close'].shift(-1)  # Next day's closing price as the target

    # Handle NaN values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Drop the last row from X to match y's length
    X = X[:-1]
    y = y.dropna()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, imputer, scaler, feature_columns


# In[36]:


def train_model(X, y):
    """
    Train a LinearRegression model on the provided data.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a LinearRegression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# In[37]:


def predict_and_decide(symbols, model, imputer, scaler, feature_columns, file_path='stock_data.csv'):
    """
    Predict the next day's stock price for each symbol and decide whether to buy, sell, or hold.
    """
    # Load the stock data
    stock_data = pd.read_csv(file_path)

    decisions = []

    for symbol in symbols:
        # Filter the data for the given symbol
        symbol_data = stock_data[stock_data['symbol'] == symbol]

        # Add features
        symbol_data = add_features(symbol_data)

        # Get the latest available data
        latest_data = symbol_data.iloc[-1]

        # Prepare the input features for prediction
        X_latest = latest_data[feature_columns].values.reshape(1, -1)

        # Handle NaN values in the input features
        X_latest = imputer.transform(X_latest)

        # Scale the input features
        X_latest = scaler.transform(X_latest)

        # Predict the next day's price
        predicted_price = model.predict(X_latest)[0]

        current_price = latest_data['close']

        # Decision-making logic
        if predicted_price > current_price * 1.02:
            decision = 'buy'
        elif predicted_price < current_price * 0.98:
            decision = 'sell'
        else:
            decision = 'hold'

        decisions.append((symbol, decision, current_price, predicted_price))

    return decisions


# In[38]:


def execute_trade(symbol, decision, hedge_sell_ratio=0.01, max_investment_ratio=0.01):
    account = alpaca.get_account()
    cash = float(account.cash)
    
    # Check if there's an existing position in the symbol
    position = alpaca.get_position(symbol) if symbol in [p.symbol for p in alpaca.list_positions()] else None
    current_qty = float(position.qty) if position else 0

    # Get the current price directly from the decisions tuple
    current_price = None
    for dec_symbol, dec, curr_price, _ in decisions:
        if dec_symbol == symbol:
            current_price = curr_price
            break

    if current_price is None:
        print(f"Current price for {symbol} not found in decisions.")
        return

    if decision == 'buy':
        # Calculate how much to buy based on available cash and hedging
        qty_to_buy = (cash * max_investment_ratio) / current_price
        alpaca.submit_order(
            symbol=symbol,
            qty=qty_to_buy,
            side='buy',
            type='market',
            time_in_force='day'
        )
    elif decision == 'sell' and current_qty > 0:
        # Calculate how much to sell, hedging by selling only a portion of current holdings
        qty_to_sell = current_qty * hedge_sell_ratio
        alpaca.submit_order(
            symbol=symbol,
            qty=qty_to_sell,
            side='sell',
            type='market',
            time_in_force='day'
        )


# ## Stock Data

# In[39]:


end_date = datetime.utcnow()
start_date = end_date - timedelta(days=30)
historical_data = get_historical_stock_data(top_50_sp500_stocks, start_date, end_date)
historical_data['date'] = pd.to_datetime(historical_data['time']).dt.date


# ## News Data

# In[40]:


news = get_news(top_50_sp500_stocks, days_prior=30)
news_df = pd.DataFrame(news)


# ## Inferences
# - **Finbert Model** for sentiment analysis based inference 
# - **Linear Regression** for stock price inference 

# In[41]:


model_name = "yiyanghkust/finbert-tone"  # Example of a financial sentiment model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# In[ ]:


news_df['sentiment'], news_df['sentiment_score'] = zip(*news_df['headline'].apply(analyze_sentiment))


# In[ ]:


news_df['created_at'] = pd.to_datetime(news_df['created_at'])

# Step 2: Extract the date (without the time) from the 'created_at' column
news_df['date'] = news_df['created_at'].dt.date

# Step 3: Explode the 'symbols' column directly since it's already a list
news_copy = news_df.explode('symbols')

# Step 4: Group by the extracted date and symbol, and calculate the average sentiment score
daily_sentiment = news_copy.groupby(['date', 'symbols'])['sentiment_score'].mean().reset_index()

# Step 5: Rename the columns for clarity
daily_sentiment.columns = ['date', 'symbol', 'average_sentiment_score']


# In[ ]:


daily_sentiment['log_sentiment_score'] = daily_sentiment['average_sentiment_score'].apply(lambda x: math.log(x))


# In[ ]:


stock_data_merged = pd.merge(historical_data, daily_sentiment, how='left', on=['date', 'symbol'])
stock_data_merged.to_csv('stock_data.csv', index=False)


# In[ ]:


update_stock_data(top_50_sp500_stocks)
stock_data = pd.read_csv("stock_data.csv")
data = pd.read_csv('stock_data.csv')


# In[ ]:


# Normalize the input features (open, high, low, close, volume, and sentiment score)
scaler = MinMaxScaler()
data[['open', 'high', 'low', 'close', 'volume', 'average_sentiment_score']] = scaler.fit_transform(
    data[['open', 'high', 'low', 'close', 'volume', 'average_sentiment_score']]
)

# Create the target variable (next day's closing price)
data['next_day_close'] = data.groupby('symbol')['close'].shift(-1)

# Drop any rows with NaN values (e.g., the last row for each symbol)
data.dropna(inplace=True)

# Split data into features and target
X = data[['open', 'high', 'low', 'close', 'volume', 'average_sentiment_score']]
y = data['next_day_close']


# In[ ]:


stock_data = pd.read_csv('stock_data.csv')


# ## Execution

# In[ ]:


X, y, imputer, scaler, feature_columns = prepare_data()

# Train the model
model = train_model(X, y)

# Get trading decisions for each symbol
decisions = predict_and_decide(top_50_sp500_stocks, model, imputer, scaler, feature_columns)


# ### Main()

# In[ ]:


def main():
    # Prepare the data
    X, y, imputer, scaler, feature_columns = prepare_data()

    # Train the model
    model = train_model(X, y)

    # Define your list of symbols
    symbols = top_50_sp500_stocks

    # Get trading decisions for each symbol
    decisions = predict_and_decide(symbols, model, imputer, scaler, feature_columns)

    # Execute trades based on the decisions
    for symbol, decision, current_price, predicted_price in decisions:
        execute_trade(symbol, decision)

if __name__ == "__main__":
    main()   # for the script to run 

