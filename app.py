import joblib
import yfinance as yf

import numpy as np

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
        pred1= self.stack1_model.predict_proba(data)[:,1]
        pred2 = self.stack2_model.predict_proba(data)[:,1]
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




import gradio as gr

def strategy(prob):
    ans =0
    alpha = 0.5  # aggressiveness scaling
    upper_th = 0.6  # 70% threshold
    lower_th = 0.4  # 30% threshold

    # Start neutral (multiplier = 1)


    # Long if above upper threshold
    if prob >= upper_th:
        ans= 1 + alpha * (prob - upper_th)

    # Short if below lower threshold
    elif prob <= lower_th:
        ans = -(1 + alpha * (lower_th - prob))
    else:
        ans=0
    return ans

# Assuming you have these classes defined somewhere
# from your_module import SentimentAnalysis, Prediction

import pandas as pd
from dash import Dash, dcc, html, Output, Input

# Assuming you have these functions/classes from your code
# from your_module import Prediction, strategy

# Load sentiment data
dictionairy = pd.read_csv("sentiment_data.csv", index_col=0)
predictions = pd.read_csv("predictions.csv", index_col=0)
# List of tickers
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'BRK.B', 'VOO', 'VTI', 'QQQ', 'SPY', 'IVV', 'JPM', 'V', 'MA', 'UNH'
]

# Prediction function
def predict_stock(ticker):
    # Stock prediction

    probs = predictions[ticker]

    # Latest prediction
    prob_up = f"🔮 Probability of going UP: {probs * 100:.2f}%"

    # Binary interpretation
    direction = "⬆️ UP" if probs > 0.5 else "⬇️ DOWN"
    direction_text = f"Predicted Direction: {direction}"

    # Sentiment analysis
    sentiment = dictionairy.loc[ticker]
    sentiment_text = (
        f"Sentiment Analysis for {ticker}: "
        f"{sentiment[0]} Positive, {sentiment[1]} Neutral, {sentiment[2]} Negative"
    )

    # Strategy output
    strat = strategy(probs)
    strat_text = strat

    return prob_up, direction_text, sentiment_text, strat_text


# Build Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("📈 Stock Price Movement Prediction"),

    dcc.Dropdown(
        id="stock-dropdown",
        options=[{"label": t, "value": t} for t in tickers],
        value="AAPL",  # default ticker
        clearable=False
    ),

    html.Div(id="prob-up", style={"marginTop": "20px", "fontSize": "18px"}),
    html.Div(id="direction", style={"marginTop": "10px", "fontSize": "18px"}),
    html.Div(id="sentiment", style={"marginTop": "10px", "fontSize": "18px"}),
    html.Div(id="strategy", style={"marginTop": "10px", "fontSize": "18px", "fontWeight": "bold"})
])


# Define callback
@app.callback(
    [Output("prob-up", "children"),
     Output("direction", "children"),
     Output("sentiment", "children"),
     Output("strategy", "children")],
    [Input("stock-dropdown", "value")]
)
def update_output(ticker):
    return predict_stock(ticker)


if __name__ == "__main__":
    app.run_server(debug=True)
















