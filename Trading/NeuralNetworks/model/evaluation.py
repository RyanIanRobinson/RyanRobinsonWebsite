import numpy as np

def evaluate_model(model, val_data):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for batch in val_data:
            inputs, targets = batch[:, :-1], batch[:, -1]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_data)

def calculate_confidence_intervals(predictions, volatility, ci_level=1.645):
    ci_upper = predictions + ci_level * volatility
    ci_lower = predictions - ci_level * volatility
    return ci_lower, ci_upper
