import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import IBMIDataset
from network import MultivariateLogisticRegression
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, num_epochs):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
    
    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            features = batch['features']
            labels = batch['labels']
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')
        avg_loss = epoch_loss / len(self.train_loader)
        print(f'Epoch [{epoch+1}/{self.num_epochs}] completed. Average Loss: {avg_loss:.4f}')
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
    
    def _validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features']
                labels = batch['labels']
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(self.val_loader)
        print(f'Validation after Epoch [{epoch+1}/{self.num_epochs}]: Average Loss: {avg_val_loss:.4f}')

    def train(self):
        for epoch in range(self.num_epochs):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)

def main():
    # Create dataset and dataloaders
    dataset = IBMIDataset(csv_file=CSV_FILE)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Initialize model, loss function, and optimizer
    input_dim = dataset.features.shape[1]
    output_dim = dataset.labels.shape[1]
    model = MultivariateLogisticRegression(input_dim=input_dim, output_dim=output_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Training loop
    trainer = Trainer(model, train_loader, criterion, optimizer, NUM_EPOCHS)
    trainer.train()

if __name__ == "__main__":
    
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DATA_DIR = r"data"
    MODEL_DIR = r"models"
    CSV_FILE = os.path.join(DATA_DIR, "nhanes_merged_complete.csv")
    MODEL_SAVE_PATH = os.path.join(DATA_DIR, "multivariate_logistic_regression.pth")
    LOG_INTERVAL = 1

    main()