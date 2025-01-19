import pandas as pd

from data_processing import download_data, clean_data
from features import calculate_technical_indicators
from model import train_model, make_predictions
from trader import make_trading_decision
from logger import setup_logger, log_training_info, log_error

def main():
    setup_logger()

    try:
        log_training_info("Starting data download...")
        raw_data = download_data(['AAPL'], '2021-01-01', '2023-12-31')

        log_training_info("Cleaning data...")
        cleaned_data = clean_data(raw_data)

        log_training_info("Calculating technical indicators...")
        features = calculate_technical_indicators(cleaned_data)

        log_training_info("Training model...")
        model = train_model(features)

        log_training_info("Predicting future prices...")
        future_predictions = make_predictions(model, features, future_days=5)

        # Log predictions
        log_training_info(f"Predictions: {future_predictions}")

        print(f"Future predictions:\n{future_predictions}")

        log_training_info("Making trading decisions...")
        current_price = cleaned_data['Close'].iloc[-1]
        make_trading_decision(future_predictions, current_price)

    except Exception as e:
        log_error(f"Error occurred: {e}")
        raise

if __name__ == '__main__':
    main()