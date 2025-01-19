import pandas as pd

def extract_features(df):
    # Example feature extraction: scaling system loads, weather, fuel costs, and reserve margins
    df['system_load_scaled'] = (df['system_load'] - df['system_load'].mean()) / df['system_load'].std()
    df['weather_scaled'] = (df['temperature'] - df['temperature'].mean()) / df['temperature'].std()
    df['fuel_cost_scaled'] = (df['fuel_cost'] - df['fuel_cost'].mean()) / df['fuel_cost'].std()
    df['reserve_margin_scaled'] = (df['reserve_margin'] - df['reserve_margin'].mean()) / df['reserve_margin'].std()
    
    # Example: Combine geography, system loads, weather, etc., into a feature matrix
    features = df[['geography', 'system_load_scaled', 'weather_scaled', 'fuel_cost_scaled', 'reserve_margin_scaled']]
    target = df['energy_price']
    
    return features, target