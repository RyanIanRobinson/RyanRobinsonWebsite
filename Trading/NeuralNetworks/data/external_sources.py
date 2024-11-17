import pandas as pd

# doing XOM - US crude oil prices easiest to find https://www.marketwatch.com/investing/stock/xom/download-data?startDate=1/4/1971&endDate=11/15/2024
def fetch_stock_data(file_path='data/raw/stock_data.csv'):
    return pd.read_csv(file_path)

# olklahoma (idk how to spell that) crude oil prices
def fetch_oil_prices(file_path='data/raw/oil_prices.csv'):
    return pd.read_csv(file_path)

# https://fred.stlouisfed.org/series/FPCPITOTLZGUSA
def fetch_inflation_rate(file_path='data/raw/inflation_data.csv'):
    return pd.read_csv(file_path)

def fetch_weather_data(file_path='data/raw/weather_data.csv'):
    return pd.read_csv(file_path)

# https://public.emdat.be/data technological and natural since 2000
def fetch_disaster_data(file_path='data/raw/disaster_data.csv'):
    return pd.read_csv(file_path)

def fetch_all_data():
    stock_data = fetch_stock_data()
    oil_data = fetch_oil_prices()
    inflation_data = fetch_inflation_rate()
    weather_data = fetch_weather_data()
    disaster_data = fetch_disaster_data()
    return stock_data, oil_data, inflation_data, weather_data, disaster_data
