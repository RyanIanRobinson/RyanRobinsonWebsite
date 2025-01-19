import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_model(model, df):
    features = df[['SMA_20', 'SMA_50', 'RSI']]  # Example features
    target = df['Close'].shift(-1)  # Predict next day's close price
    
    features = features[:-1]  # Drop last row for alignment
    target = target[:-1]  # Drop last row for alignment

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    # Print predictions vs. actual values
    print("Predictions vs Actual:")
    for actual, pred in zip(y_test, y_pred):
        print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

    # Save evaluation metrics and predictions to a file
    with open('evaluation.txt', 'w') as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write("Predictions vs Actual:\n")
        for actual, pred in zip(y_test, y_pred):
            f.write(f"Actual: {actual:.2f}, Predicted: {pred:.2f}\n")

def evaluate_with_cross_validation(model, df):
    features = df[['SMA_20', 'SMA_50', 'RSI']]
    target = df['Close'].shift(-1)

    features = features[:-1]
    target = target[:-1]

    scores = cross_val_score(model, features, target, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores
    avg_mse = np.mean(mse_scores)

    print(f"Cross-Validation MSE Scores: {mse_scores}")
    print(f"Average MSE: {avg_mse}")

    with open('evaluation_cv.txt', 'w') as f:
        f.write(f"Cross-Validation MSE Scores: {mse_scores}\n")
        f.write(f"Average MSE: {avg_mse}\n")
