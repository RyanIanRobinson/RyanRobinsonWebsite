import pandas as pd
import yfinance as yf
import os
from alpha_vantage.timeseries import TimeSeries

def download_data(symbols, start_date, end_date, source='yahoo'):
    if source == 'yahoo':
        data = {}
        for symbol in symbols:
            df = yf.download(symbol, start=start_date, end=end_date)
            df['Symbol'] = symbol
            data[symbol] = df
        return pd.concat(data.values(), keys=data.keys())
    elif source == 'alpha_vantage':
        api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Replace with your actual API key
        ts = TimeSeries(key=api_key, output_format='pandas')
        data = {}
        for symbol in symbols:
            df, _ = ts.get_daily(symbol=symbol, outputsize='full')
            df['Symbol'] = symbol
            data[symbol] = df
        return pd.concat(data.values(), keys=data.keys())
    else:
        raise ValueError("Unsupported data source.")

def save_raw_data(data, filename):
    data.to_csv(filename)

def clean_data(df):
    df = df.dropna()
    df = df[df['Volume'] > 0]
    return df

def save_processed_data(df, filename):
    filepath = os.path.join('data', filename)
    df.to_csv(filepath)
