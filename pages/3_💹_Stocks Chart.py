# Updated Historical Data and Technical Indicators Tables
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# Set up Streamlit page layout
st.set_page_config(
    page_title="Stocks Dashboard 💹",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and header
st.title('📈 Stocks Dashboard')
st.markdown("---")

# Fetch stock data based on the ticker, period, and interval
def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    
    if data.empty:
        return None  # Return `None` to indicate no data
    
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(key=ticker, axis=1, level='Ticker')  # Select the correct ticker level
    
    return data

# Process data to ensure it is timezone-aware and has the correct format
def process_data(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

# Calculate basic metrics from the stock data
def calculate_metrics(data):
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()
    return last_close, change, pct_change, high, low, volume

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    data['Bollinger_Upper'] = sma + (std_dev * num_std)
    data['Bollinger_Lower'] = sma - (std_dev * num_std)
    return data

# Add technical indicators
def add_technical_indicators_manual(data):
    if len(data) < 20:
        st.warning("Not enough data to calculate technical indicators. Returning raw data.")
        return data

    # Simple Moving Average (SMA)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    # Exponential Moving Average (EMA)
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    data['RSI_14'] = calculate_rsi(data)

    # Bollinger Bands
    data = calculate_bollinger_bands(data)

    return data

# Sidebar for user input parameters
st.sidebar.header('📊 Chart Settings')
ticker = st.sidebar.text_input('Ticker Symbol', 'ADBE')
time_period = st.sidebar.selectbox('Time Period', ['1d', '1wk', '1mo', '1y', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20', 'RSI_14', 'Bollinger Bands'])

# Mapping of time periods to data intervals
interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

# Automatically fetch and display data when the app loads or when parameters change
data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
if data is not None:
    data = process_data(data)
    data = add_technical_indicators_manual(data)
    
    last_close, change, pct_change, high, low, volume = calculate_metrics(data)
    
    # Display main metrics
    st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} USD", delta=f"{change:.2f} ({pct_change:.2f}%)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("High", f"{high:.2f} USD")
    col2.metric("Low", f"{low:.2f} USD")
    col3.metric("Volume", f"{volume:,}")
    
    # Plot the stock price chart
    fig = go.Figure()
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(x=data['Datetime'],
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     increasing_line_color='#00FF00',  # Green for up
                                     decreasing_line_color='#FF0000'))  # Red for down
    else:
        fig = px.line(data, x='Datetime', y='Close', color_discrete_sequence=['#6200EA'])  # Purple line
    
    # Add selected technical indicators to the chart
    for indicator in indicators:
        if indicator == 'SMA 20':
            fig.add_trace(go.Scatter(
                x=data['Datetime'], 
                y=data['SMA_20'], 
                line=dict(color='#00FF00'),  # Green
                name='SMA 20'
            ))
        elif indicator == 'EMA 20':
            fig.add_trace(go.Scatter(
                x=data['Datetime'], 
                y=data['EMA_20'], 
                line=dict(color='#FFA500'),  # Orange
                name='EMA 20'
            ))
        elif indicator == 'Bollinger Bands':
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data['Bollinger_Upper'],
                name='Bollinger Upper',
                line=dict(color='#00FFFF', dash='dash')  # Cyan
            ))
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data['Bollinger_Lower'],
                name='Bollinger Lower',
                line=dict(color='#00FFFF', dash='dash')  # Cyan
            ))
    
    # Format graph
    fig.update_layout(
        title=f'{ticker} {time_period.upper()} Chart',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        height=600,
        #plot_bgcolor='#1E1E1E',  # Dark background
        #paper_bgcolor='#1E1E1E',  # Dark background
        font=dict(color='#E0E0E0')  # Light text
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot the RSI chart (in a separate figure)
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=data['Datetime'],
        y=data['RSI_14'],
        name='RSI 14',
        line=dict(color='#6200EA')  # Purple
    ))
    rsi_fig.add_trace(go.Scatter(
        x=data['Datetime'],
        y=[30] * len(data),
        name='RSI 30',
        line=dict(color='#00FF00', dash='dash')  # Green
    ))
    rsi_fig.add_trace(go.Scatter(
        x=data['Datetime'],
        y=[70] * len(data),
        name='RSI 70',
        line=dict(color='#FF0000', dash='dash')  # Red
    ))
    rsi_fig.update_layout(
        title=f'{ticker} RSI (14)',
        xaxis_title='Time',
        yaxis_title='RSI',
        height=300,
        showlegend=False,
        #plot_bgcolor='#1E1E1E',  # Dark background
        #paper_bgcolor='#1E1E1E',  # Dark background
        font=dict(color='#E0E0E0')  # Light text
    )
    st.plotly_chart(rsi_fig, use_container_width=True)

    # Display historical data and technical indicators
    st.subheader('📊 Historical Data')
    with st.expander("View Historical Data", expanded=False):
        st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].style.format({
            'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 'Close': '{:.2f}', 'Volume': '{:,}'
        }).applymap(lambda x: 'color: #E0E0E0').set_properties(**{
            'background-color': '#1E1E1E',
            'border': '1px solid #333333'
        }))
    
    st.subheader('📈 Technical Indicators')
    with st.expander("View Technical Indicators", expanded=False):
        st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20', 'RSI_14', 'Bollinger_Upper', 'Bollinger_Lower']].style.format({
            'SMA_20': '{:.2f}', 'EMA_20': '{:.2f}', 'RSI_14': '{:.2f}', 'Bollinger_Upper': '{:.2f}', 'Bollinger_Lower': '{:.2f}'
        }).applymap(lambda x: 'color: #E0E0E0').set_properties(**{
            'background-color': '#1E1E1E',
            'border': '1px solid #333333'
        }))

else:
    st.error(f"No data found for ticker: {ticker}. Please check the ticker symbol and try again.")

# Sidebar section for real-time stock prices of selected symbols
st.sidebar.header('📈 Real-Time Stock Prices')
stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
for symbol in stock_symbols:
    real_time_data = fetch_stock_data(symbol, '1d', '1m')
    if not real_time_data.empty:
        real_time_data = process_data(real_time_data)
        last_price = real_time_data['Close'].iloc[-1]
        change = last_price - real_time_data['Open'].iloc[0]
        pct_change = (change / real_time_data['Open'].iloc[0]) * 100
        st.sidebar.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")

# Sidebar information section
st.sidebar.subheader('ℹ️ About')
st.sidebar.info('This dashboard provides stock data and technical indicators for various time periods. Use the sidebar to customize your view.')