from data_processing.load_data import collect_data, clean_data
from data_processing.feature_engineering import extract_features
from model.neural_network import train_model
from model.predict import make_predictions
from evaluation.evaluate import evaluate_model
from logger.logger import setup_logger, log_info, log_error
import requests
import pandas as pd

openweathermap_api_key = 'df306d129f156c114ba9452da546ae10'
city = 'San Francisco'

eia_api_key = "YlVp5gTVgSaaXMko5VaDLG4bepKVRkp9j53Pfhl6"

def main():
    setup_logger()

    url = f"https://api.eia.gov/v2/electricity/rto/wholesale-prices/data/?api_key={eia_api_key}"
    response = requests.get(url)
    data = response.json()
    series_data = data['response']['data']
    df = pd.DataFrame(series_data, columns=['date', 'value'])
    df.columns = ['Date', 'System_Load']
    print("testing system_load\n\n\n", data, "\n\n\ntested system_load")
    
    try:
        log_info("Loading data...")
        df = collect_data(openweathermap_api_key, eia_api_key, city)
        print("\nLoaded Data:\n", df.head())
        
        log_info("Cleaning data...")
        df = clean_data(df)
        print("\nCleaned Data:\n", df.columns, "\n", df)

        log_info("Extracting features...")
        features, target = extract_features(df)
        print("\nExtracted features:\n", features.columns, "\n", features)
        print("\nExtracted target:\n", target.columns, "\n", target)

        log_info("Training model...")
        model = train_model(features, target)
        print("\nTrained Model:\n", model, "\n")

        log_info("Evaluating model...")
        evaluate_model(model, features, target)
        print("\nEvaluated Model", "\n")

        log_info("Making predictions...")
        predictions = make_predictions(model, features)
        print("\nPredictions:\n", predictions)

    except Exception as e:
        log_error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()
