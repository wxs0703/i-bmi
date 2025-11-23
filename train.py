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
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
    
    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            features = batch['features']
            labels = batch['labels']
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            acc = (outputs.round() == labels).float().mean()
            epoch_acc += acc.item()
            self.optimizer.step()
            epoch_loss += loss.item()
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}')
        avg_loss = epoch_loss / len(self.train_loader)
        avg_acc = epoch_acc / len(self.train_loader)
        print(f'Epoch [{epoch+1}/{self.num_epochs}] completed. Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}')
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)

        return avg_loss, avg_acc
    
    def _validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features']
                labels = batch['labels']
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                acc = (outputs.round() == labels).float().mean()
                val_acc += acc.item()
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_acc = val_acc / len(self.val_loader)
        print(f'Validation after Epoch [{epoch+1}/{self.num_epochs}]: Average Loss: {avg_val_loss:.4f}, Average Accuracy: {avg_val_acc:.4f}')

        return avg_val_loss, avg_val_acc

    def train(self):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._validate_epoch(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
        # Save training loss to CSV
        loss_df = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accuracies,
            'val_accuracy': val_accuracies
        })
        loss_df.to_csv(LOSS_SAVE_PATH, index=False)

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
    trainer = Trainer(model, train_loader, test_loader,criterion, optimizer, NUM_EPOCHS)
    trainer.train()

if __name__ == "__main__":
    
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DATA_DIR = r"data"
    MODEL_DIR = r"models"
    CSV_FILE = os.path.join(DATA_DIR, "nhanes_merged_complete.csv")
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "multivariate_logistic_regression.pth")
    LOSS_SAVE_PATH = os.path.join(MODEL_DIR, "training_loss.csv")
    LOG_INTERVAL = 20

    main()