import pandas as pd
import requests

def get_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    return data

# Example usage
api_key = 'df306d129f156c114ba9452da546ae10'
city = 'San Francisco'


def clean_data(df):
    # Clean the data (handling missing values, filtering outliers, etc.)
    df = df.dropna()  # Example: drop rows with NaN
    return df