from sklearn.metrics import mean_squared_error

def evaluate_model(model, features, target):
    predictions = model.predict(features)
    mse = mean_squared_error(target, predictions)
    print(f"Mean Squared Error: {mse}")