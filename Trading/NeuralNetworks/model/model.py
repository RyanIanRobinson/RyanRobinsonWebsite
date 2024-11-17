import torch
import torch.nn as nn

class LSTMTransformerPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMTransformerPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=4, num_encoder_layers=2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        transformed_out = self.transformer(lstm_out)
        out = self.fc(transformed_out[:, -1, :])
        return out
