import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import product
from datetime import datetime, timedelta
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Ignore warnings
warnings.filterwarnings("ignore")

# Streamlit configuration
st.set_page_config(
    page_title="Crypto Price Prediction",
    page_icon="📈",
    layout="wide",
)

st.title("Cryptocurrency Price Prediction 📈")

# Sidebar for user input parameters
st.sidebar.header("Price Prediction Parameters")
ticker = st.sidebar.text_input("Ticker", "AAPL")
time_period = st.sidebar.selectbox(
    "Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "max"], index=1
)
prediction_ahead = st.sidebar.number_input(
    "Prediction Days Ahead", min_value=1, max_value=30, value=7, step=1
)
forecasting_method = st.sidebar.selectbox(
    "Forecasting Method",
    ["ARIMA", "Prophet", "Linear Regression", "Exponential Smoothing"],
    index=0,
)
chart_type = st.sidebar.selectbox(
    "Chart Type", ["Line", "Candlestick", "Heikin-Ashi Candlestick"], index=0
)

# Fetch stock data
def fetch_stock_data(ticker, time_period):
    interval_mapping = {
        "1mo": "1h",
        "3mo": "1d",
        "6mo": "1d",
        "1y": "1wk",
        "2y": "1wk",
        "max": "1mo",
    }
    interval = interval_mapping.get(time_period, "1d")
    data = yf.download(ticker, period=time_period, interval=interval, group_by="ticker")
    
    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(key=ticker, axis=1, level="Ticker")

    return data

# ARIMA model for predictions
def predict_arima(data, prediction_ahead):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    p_values, d_values, q_values = range(0, 3), range(0, 2), range(0, 3)

    def evaluate_arima_model(train, test, arima_order):
        try:
            model = ARIMA(train["Close"], order=arima_order)
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            mse = mean_squared_error(test["Close"], predictions)
            return mse, model_fit
        except:
            return float("inf"), None

    results = []
    for p, d, q in product(p_values, d_values, q_values):
        mse, model_fit = evaluate_arima_model(train, test, (p, d, q))
        results.append(((p, d, q), mse, model_fit))

    best_order, _, best_model = min(results, key=lambda x: x[1])
    forecast = best_model.forecast(steps=len(test) + prediction_ahead)
    return forecast, pd.date_range(
        start=data.index[-1], periods=prediction_ahead + 1, freq="D"
    )[1:]

# Prophet model for predictions
def predict_prophet(data, prediction_ahead):
    df = data.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=prediction_ahead)
    forecast = model.predict(future)

    return forecast

# Regression-based time series prediction
def predict_regression(data, prediction_ahead):
    df = data.reset_index()
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days
    X = df[["Days"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    # Generate future dates
    future_days = np.arange(X.iloc[-1, 0] + 1, X.iloc[-1, 0] + 1 + prediction_ahead)
    future_dates = pd.date_range(
        start=data.index[-1], periods=prediction_ahead + 1, freq="D"
    )[1:]

    # Predict future prices
    future_prices = model.predict(future_days.reshape(-1, 1))
    return future_prices, future_dates

# Exponential Smoothing model for predictions
def predict_exponential_smoothing(data, prediction_ahead):
    model = ExponentialSmoothing(data["Close"], trend="add", seasonal="add", seasonal_periods=12)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=prediction_ahead)
    future_dates = pd.date_range(start=data.index[-1], periods=prediction_ahead + 1, freq="D")[1:]
    return forecast, future_dates

# Heikin-Ashi Candlestick Calculation
def calculate_heikin_ashi(data):
    ha_data = data.copy()
    ha_data["HA_Close"] = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4
    ha_data["HA_Open"] = 0.0
    ha_data["HA_High"] = 0.0
    ha_data["HA_Low"] = 0.0

    for i in range(len(ha_data)):
        if i == 0:
            ha_data.iloc[i, ha_data.columns.get_loc("HA_Open")] = data.iloc[i]["Open"]
        else:
            ha_data.iloc[i, ha_data.columns.get_loc("HA_Open")] = (
                ha_data.iloc[i - 1]["HA_Open"] + ha_data.iloc[i - 1]["HA_Close"]
            ) / 2

        ha_data.iloc[i, ha_data.columns.get_loc("HA_High")] = max(
            data.iloc[i]["High"],
            ha_data.iloc[i]["HA_Open"],
            ha_data.iloc[i]["HA_Close"],
        )
        ha_data.iloc[i, ha_data.columns.get_loc("HA_Low")] = min(
            data.iloc[i]["Low"],
            ha_data.iloc[i]["HA_Open"],
            ha_data.iloc[i]["HA_Close"],
        )

    return ha_data

# Main logic
if st.sidebar.button("Predict"):
    with st.spinner("Fetching data and training the model..."):
        data = fetch_stock_data(ticker, time_period)
        if data is None:
            st.error("No data found for the selected ticker and time period.")
        else:
            if forecasting_method == "ARIMA":
                predictions, future_dates = predict_arima(data, prediction_ahead)
            elif forecasting_method == "Prophet":
                predictions = predict_prophet(data, prediction_ahead)
            elif forecasting_method == "Linear Regression":
                predictions, future_dates = predict_regression(data, prediction_ahead)
            elif forecasting_method == "Exponential Smoothing":
                predictions, future_dates = predict_exponential_smoothing(data, prediction_ahead)

            # Visualization
            fig = go.Figure()

            if chart_type == "Candlestick":
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data["Open"],
                        high=data["High"],
                        low=data["Low"],
                        close=data["Close"],
                        name="Historical Data",
                    )
                )
            elif chart_type == "Heikin-Ashi Candlestick":
                ha_data = calculate_heikin_ashi(data)
                fig.add_trace(
                    go.Candlestick(
                        x=ha_data.index,
                        open=ha_data["HA_Open"],
                        high=ha_data["HA_High"],
                        low=ha_data["HA_Low"],
                        close=ha_data["HA_Close"],
                        name="Heikin-Ashi Historical Data",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data["Close"],
                        mode="lines",
                        name="Historical Prices",
                    )
                )

            if forecasting_method == "Prophet":
                fig.add_trace(
                    go.Scatter(
                        x=predictions["ds"],
                        y=predictions["yhat"],
                        mode="lines",
                        name="Prophet Predictions",
                        line=dict(color="green", dash="dot"),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=predictions[-prediction_ahead:],
                        mode="lines",
                        name=f"{forecasting_method} Predictions",
                        line=dict(color="orange", dash="dash"),
                    )
                )

            fig.update_layout(
                title=f"{ticker.upper()} Price Prediction ({time_period})",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=600,
                legend_title="Legend",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display metrics
            last_known_price = data["Close"].iloc[-1]
            st.metric("Last Known Price", f"${last_known_price:,.2f}")
            if forecasting_method == "Prophet":
                st.metric(
                    f"Prophet Predicted Price After {prediction_ahead} Days",
                    f"${predictions['yhat'].iloc[-1]:,.2f}",
                )
            else:
                st.metric(
                    f"{forecasting_method} Predicted Price After {prediction_ahead} Days",
                    f"${predictions[-1]:,.2f}",
                )