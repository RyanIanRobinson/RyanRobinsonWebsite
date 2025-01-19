import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output: predicting energy prices
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(features, target):
    model = build_model(features.shape[1])
    model.fit(features, target, epochs=100, batch_size=32, validation_split=0.2)
    model.save('model/energy_price_predictor.h5')  # Save the model
    return model