import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import numpy as np
import praw
from collections import Counter
import re
import numpy as np
from transformers import pipeline
# Load trained model


class Prediction:
    def __init__(self):
        self.meta_model = joblib.load("meta_model.pkl")
        self.stack1_model = joblib.load("stackmodel1.pkl")
        self.stack2_model = joblib.load("stackmodel2.pkl")
        self.DataCleaner = DataCleaner()
    def call(self, ticker):
        df = yf.download(ticker, period="12mo", interval="1d")
        df.columns = ["Close", "High", "Low", "Open", "Volume"]
        data = DataCleaner().call(df)
        data = data.rename(columns={"Close": "Adj Close"})
        data = data.drop("target", axis=1)
        n = len(data) //2
        pred1= self.stack1_model.predict_proba(data[:n])[:,1]
        pred2 = self.stack2_model.predict_proba(data[n:])[:,1]
        stacked_X = np.column_stack((pred1, pred2))
        final_pred = self.meta_model.predict_proba(stacked_X)
        return final_pred[:,1]

class DataCleaner:
    def call(self,df):
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

        # Exponential Moving Average (EMA)
        df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        rolling_mean = df["Close"].rolling(window=20).mean()
        rolling_std = df["Close"].rolling(window=20).std()

        df["Bollinger_Upper"] = rolling_mean + (rolling_std * 2)
        df["Bollinger_Lower"] = rolling_mean - (rolling_std * 2)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0,  0.0)
        loss = -delta.where(delta < 0, 0.0)

        roll_up = pd.Series(gain).rolling(14).mean()
        roll_down = pd.Series(loss).rolling(14).mean()

        RS = roll_up / roll_down
        df["RSI"] = 100 - (100 / (1 + RS))
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()

        df["MACD"] = ema12 - ema26
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["OBV"] = obv

        for lag in range(1, 12):
            #df[f'Close_lag{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag{lag}'] = df['Volume'].shift(lag)
            df[f'Signal_lag{lag}'] = df['Signal'].shift(lag)
            df[f'OBV_lag{lag}'] = df['OBV'].shift(lag)
            df[f'Log_retrun_lag{lag}'] = df['LogReturn'].shift(lag)
            df[f'RSI_lag{lag}'] = df['RSI'].shift(lag)
        df = df.drop(["Open","High","Low"], axis =1)

        return df

class SentimentAnalysis:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis",
                     model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True,max_length=512)
        self.reddit = praw.Reddit(
    client_id="4ioSwGbN_zpCSNxIgtlv0g",
    client_secret="iQgdmxeV39vWD9_Fl-AKQFmPa52G9Q",
    user_agent="python:science_analysis_bot:v1.0 (by u/datwhiteguynum2)"
)
    def sentiment_picker(self,sentiment):
        val = {"positive":[1,0,0],"negative":[0,0,1],"neutral":[0,1,0]}
        return np.array(val[sentiment["label"]])

    def calculate_score(self,dictionairy,sentiment):
        dictionairy[0] += self.sentiment_picker(sentiment)
        return None

    def call(self, subreddits):
        dictionairy = {}
        common_tickers = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                  'BRK.B', 'VOO', 'VTI', 'QQQ', 'SPY', 'IVV', 'JPM', 'V', 'MA', 'UNH'}

        # Pattern to find UPPER CASE words that might be tickers (3-5 letters)
        ticker_pattern = re.compile(r'\b[A-Z]{3,5}\b')
        reddits =[]

        ticker_counter = Counter()
        for i in range(len(subreddits)):
            reddits += list(self.reddit.subreddit(subreddits[i]).top(time_filter="week", limit=500))

        for submission in reddits:
            title_text = submission.title + " " + (submission.selftext or "")
            potential_tickers = ticker_pattern.findall(title_text)
            for ticker in potential_tickers:
                if ticker in common_tickers:
                    ticker_counter[ticker] += 1
                    sentiment = self.classifier(submission.selftext or "")[0]
                    if ticker_counter[ticker] <= 1:
                        dictionairy[ticker] = [self.sentiment_picker(sentiment)]
                    else:
                        self.calculate_score(dictionairy[ticker], sentiment)

            # Check comments
            submission.comments.replace_more(limit=10) # Limit comment exploration
            for comment in submission.comments.list():
                comment_text = comment.body
                potential_tickers = ticker_pattern.findall(comment_text)
                for ticker in potential_tickers:
                    if ticker in common_tickers:
                        ticker_counter[ticker] += 1
                        sentiment = self.classifier(submission.selftext or "")[0]
                        if ticker_counter[ticker] <= 1:
                            dictionairy[ticker] = [self.sentiment_picker(sentiment)]
                        else:
                            self.calculate_score(dictionairy[ticker], sentiment)
        return dictionairy


st.title("📈 Stock Price Movement Prediction")

ticker = st.selectbox(
    "Choose a stock:",
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                  'BRK.B', 'VOO', 'VTI', 'QQQ', 'SPY', 'IVV', 'JPM', 'V', 'MA', 'UNH']  # you can add more here
)
dictionairy = SentimentAnalysis().call("r/investing","r/stocks","r/wallstreetbets","r/personalfinance","r/FinancialPlanning","r/financialindependence","r/CryptoCurrency")
if st.button("Predict"):
    model = Prediction()
    probs = model.call(ticker)


    # Show last prediction
    st.subheader(f"Latest prediction for {ticker}")
    st.write(f"🔮 Probability of going UP: {probs[-1]*100:.2f}%")

    # Add binary interpretation
    direction = "⬆️ UP" if probs[-1] > 0.5 else "⬇️ DOWN"
    st.write(f"Predicted Direction: **{direction}**")

    st.write(f"Sentiment Analysis for {ticker}: {dictionairy[ticker][0]} Positive {dictionairy[ticker][0]} Neutral {dictionairy[ticker][0]} Negative")
    direction = "⬆️ UP" if probs[-1] > 0.5 else "⬇️ DOWN"
    st.write(f"Predicted Direction: **{direction}**")


