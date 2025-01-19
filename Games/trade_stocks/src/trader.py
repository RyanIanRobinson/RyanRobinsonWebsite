import pandas as pd

def make_trading_decision(predictions, current_price):
    last_prediction = predictions.iloc[-1]
    
    if last_prediction['Predicted_Close_Price'] > current_price:
        decision = "Buy"
    elif last_prediction['Predicted_Close_Price'] < current_price:
        decision = "Sell"
    else:
        decision = "Hold"

    print(f"Trading Decision: {decision}")
    return decision
