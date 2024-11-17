import pandas as pd

def preprocess(data):
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    data = data.astype(float)  # Ensuring all data is numeric
    return data

def preprocess_all_data(stock_data, oil_data, inflation_data, weather_data, disaster_data):
    stock_data = preprocess(stock_data)
    oil_data = preprocess(oil_data)
    inflation_data = preprocess(inflation_data)
    weather_data = preprocess(weather_data)
    disaster_data = preprocess(disaster_data)
    merged_data = pd.concat([stock_data, oil_data, inflation_data, weather_data, disaster_data], axis=1)
    return merged_data.dropna()
