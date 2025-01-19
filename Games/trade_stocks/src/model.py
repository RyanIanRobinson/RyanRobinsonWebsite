import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
import pandas as pd

def train_model(df):
    features = df[['SMA_20', 'SMA_50', 'RSI', 'Close']]
    target = df['Close'].shift(-1)

    features = features[:-1]
    target = target[:-1]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')

    return model

def calculate_confidence_intervals(sim_results):
    # Helper function to calculate confidence intervals from simulation results.
    return {
        'CI_Lower_95%': np.percentile(sim_results, 2.5),
        'CI_Upper_95%': np.percentile(sim_results, 97.5),
        'CI_Lower_90%': np.percentile(sim_results, 5),
        'CI_Upper_90%': np.percentile(sim_results, 95),
        'CI_Lower_80%': np.percentile(sim_results, 10),
        'CI_Upper_80%': np.percentile(sim_results, 90),
    }

def make_predictions(model, df, future_days=5, num_simulations=100):
    # Extract the relevant features (e.g., SMA_20, SMA_50, RSI) for prediction
    future_features = df[['SMA_20', 'SMA_50', 'RSI', 'Close']].iloc[-1:]  # Use the latest row for future predictions

    # Get the current stock price for reference
    current_price = df['Close'].iloc[-1]

    # Calculate the historical volatility (based on percentage change of close prices)
    df['Returns'] = df['Close'].pct_change()
    volatility = df['Returns'].rolling(window=20).std().iloc[-1]  # 20-day historical volatility

    # Initialize lists to store predictions and confidence intervals
    future_df = pd.DataFrame(columns=['Day', 'Predicted_Close_Price', 'CI_Lower_95%', 'CI_Upper_95%', 
                                      'CI_Lower_90%', 'CI_Upper_90%', 'CI_Lower_80%', 'CI_Upper_80%'])

    # Simulate predictions for each future day
    for day in range(1, future_days + 1):
        sim_results = []

        # Run multiple simulations to get a range of possible outcomes
        for _ in range(num_simulations):
            # Predict the close price
            pred = model.predict(future_features)[0]

            # Add noise based on historical volatility to simulate future price uncertainty
            noise = np.random.normal(0, pred * volatility) 
            sim_results.append(pred + noise)

        # Calculate the mean prediction and confidence intervals
        mean_pred = np.mean(sim_results)
        ci_intervals = calculate_confidence_intervals(sim_results)

        # Create a DataFrame for future predictions
        row_df = pd.DataFrame({
                'Day': [day],
                'Predicted_Close_Price': [mean_pred],
                **ci_intervals
            })
        
        future_df = pd.concat([future_df, row_df], ignore_index=True)

    # Print current price for comparison
    print(f"Current Close Price: {current_price}")
    
    return future_df

def calculate_rsi(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
