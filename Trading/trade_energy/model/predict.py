import tensorflow as tf
import pandas as pd

def make_predictions(model, features):
    predictions = model.predict(features)
    return predictions