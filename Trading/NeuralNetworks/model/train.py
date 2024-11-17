import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, data, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_data, val_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]

    for epoch in range(epochs):
        model.train()
        for batch in train_data:
            inputs, targets = batch[:, :-1], batch[:, -1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            val_loss = evaluate_model(model, val_data)
            print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), 'calculations/model_weights.pth')
    print("Model trained and saved successfully.")
