import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SketchNet
from dataset import SketchifyDataset
import os

# Load Dataset
dataset_path = r"training\BSR\BSDS500\images\train"

train_dataset = SketchifyDataset(dataset_path)
train_loader = DataLoader(train_dataset,
                          batch_size=8,
                          shuffle=True)

# Initialize Model
model = SketchNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=1e-3)

# Train
epochs = 20
for epoch in range(epochs):
    total_loss = 0

    for X, y in train_loader:
        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch: {epoch + 1} / {epochs}, Loss: {total_loss:.4f}")

# Save model
os.makedirs("sketchify/backend/Models", exist_ok=True)
torch.save(model.state_dict(), "sketchify/backend/Models/sketchify_cnn.pt")
print("Model saved!")