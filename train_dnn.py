import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from edited_bmi_data import IBMIDataset
from deep_neural_network import DeepNeuralNetwork, DeepResidualNetwork
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class DNNTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 num_epochs, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.best_val_loss = float('inf')
    
    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)

            loss = self.criterion(outputs, labels)
       
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            acc = (outputs.round() == labels).float().mean()
            epoch_acc += acc.item()
            
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                      f'Step [{batch_idx+1}/{len(self.train_loader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}')
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_acc = epoch_acc / len(self.train_loader)
        
        print(f'Epoch [{epoch+1}/{self.num_epochs}] completed. '
              f'Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}')
        
        return avg_loss, avg_acc
    
    def _test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                acc = (outputs.round() == labels).float().mean()
                test_acc += acc.item()
        
        avg_test_loss = test_loss / len(self.val_loader)
        avg_test_acc = test_acc / len(self.val_loader)
        
        print(f'test after Epoch [{epoch+1}/{self.num_epochs}]: '
              f'Average Loss: {avg_test_loss:.4f}, Average Accuracy: {avg_test_acc:.4f}')
        
        if avg_test_loss < self.best_val_loss:
            self.best_val_loss = avg_test_loss
            torch.save(self.model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f'  --> Best model saved with test loss: {avg_test_loss:.4f}')
        
        return avg_test_loss, avg_test_acc
    
    def train(self):
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_epoch(epoch)
            test_loss, test_acc = self._test_epoch(epoch)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
        
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        
        history_df = pd.DataFrame({
            'train_loss': train_losses,
            'test_loss': test_losses,
            'train_accuracy': train_accuracies,
            'test_accuracy': test_accuracies
        })
        history_df.to_csv(HISTORY_SAVE_PATH, index=False)
        
        print(f'\nTraining completed!')
        print(f'Training history saved to {HISTORY_SAVE_PATH}')
        print(f'Final model saved to {MODEL_SAVE_PATH}')
        print(f'Best model saved to {BEST_MODEL_SAVE_PATH}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    print(f'Loading data from {CSV_FILE}...')
    dataset = IBMIDataset(csv_file=CSV_FILE)
    print(f'Dataset loaded: {len(dataset)} samples')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of labels: {dataset.num_labels}')
    print(f'Label columns: {dataset.label_cols}\n')
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f'Training samples: {train_size}')
    print(f'Test samples: {test_size}\n')
    
    input_dim = dataset.num_features
    output_dim = dataset.num_labels
    
    if MODEL_TYPE == 'dnn':
        model = DeepNeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=HIDDEN_DIMS,
            dropout_rate=DROPOUT_RATE
        )
    elif MODEL_TYPE == 'resnet':
        model = DeepResidualNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=HIDDEN_DIM,
            num_blocks=NUM_BLOCKS,
            dropout_rate=DROPOUT_RATE
        )
    else:
        raise ValueError(f"MODEL_TYPE must be 'dnn' or 'resnet', got {MODEL_TYPE}")
    
    print(f'Model type: {MODEL_TYPE}')
    print('Model architecture:')
    print(model)
    print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters())}\n')
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    print('='*80)
    print('Starting training...')
    print('='*80 + '\n')
    
    trainer = DNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=device
    )
    
    trainer.train()


if __name__ == "__main__":
    MODEL_TYPE = 'dnn'  
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5 
    NUM_EPOCHS = 50
    DROPOUT_RATE = 0.2
    
    HIDDEN_DIMS = [256, 256, 128, 128, 64, 64, 32]
    HIDDEN_DIM = 128  
    NUM_BLOCKS = 3 
    
    DATA_DIR = r"data"
    MODEL_DIR = r"models"
    CSV_FILE = os.path.join(DATA_DIR, "nhanes_merged_complete.csv")
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"deep_neural_network_{MODEL_TYPE}.pth")
    BEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"best_deep_neural_network_{MODEL_TYPE}.pth")
    HISTORY_SAVE_PATH = os.path.join(MODEL_DIR, f"nn_training_history_{MODEL_TYPE}_complex.csv")
    
    LOG_INTERVAL = 20
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    main()