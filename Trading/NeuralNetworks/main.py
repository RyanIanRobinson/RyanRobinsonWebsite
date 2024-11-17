from data.external_sources import fetch_all_data
from data.preprocess import preprocess_all_data
from calculations.model import LSTMTransformerPredictor
from calculations.train import train_model
from calculations.evaluation import evaluate_model, calculate_confidence_intervals
from user_interface.visualize import plot_predictions

def main():
    # Fetch and preprocess data
    stock_data, oil_data, inflation_data, weather_data, disaster_data = fetch_all_data()
    data = preprocess_all_data(stock_data, oil_data, inflation_data, weather_data, disaster_data)

    # Model Initialization
    model = LSTMTransformerPredictor(input_size=data.shape[1], hidden_size=64, num_layers=2, output_size=1)

    # Train the Model
    train_model(model, data)

    # Evaluate the Model and Generate Predictions
    predictions, volatility = model(data)
    ci_lower, ci_upper = calculate_confidence_intervals(predictions, volatility)

    # Visualize Predictions
    plot_predictions(data['stock_price'], predictions, ci_lower, ci_upper)

if __name__ == "__main__":
    main()
