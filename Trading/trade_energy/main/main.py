from data_processing.load_data import get_weather_data, clean_data
from data_processing.feature_engineering import extract_features
from model.neural_network import train_model
from model.predict import make_predictions
from evaluation.evaluate import evaluate_model
from logger.logger import setup_logger, log_info, log_error

api_key = 'df306d129f156c114ba9452da546ae10'
city = 'San Francisco'

def main():
    setup_logger()
    
    try:
        log_info("Loading data...")
        df = get_weather_data(api_key, city)
        
        log_info("Cleaning data...")
        df = clean_data(df)

        log_info("Extracting features...")
        features, target = extract_features(df)

        log_info("Training model...")
        model = train_model(features, target)

        log_info("Evaluating model...")
        evaluate_model(model, features, target)

        log_info("Making predictions...")
        predictions = make_predictions(model, features)
        print(predictions)

    except Exception as e:
        log_error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()
