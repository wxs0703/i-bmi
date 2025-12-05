import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from data import IBMIDataset
from network import MultivariateLogisticRegression
from deep_neural_network import DeepNeuralNetwork
from torch.utils.data import DataLoader

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_accuracy_curve(train_accuracies, val_accuracies, save_path):
    plt.figure()
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy over Epochs')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def analyze_model(model, data_loader, device, output_dir):
    
    # # analyze model weights
    # weights = model.linear.weight.data.cpu().numpy()
    # num_conditions = weights.shape[0]
    # feature_names = data_loader.dataset.features.columns.tolist()
    # # take L2 norm of weights for each feature across all conditions, to get overall importance
    # weight_magnitudes = np.linalg.norm(weights, axis=0)
    # feature_importance = pd.DataFrame({
    #     'Feature': feature_names,
    #     'Weight_Magnitude': weight_magnitudes
    # }).sort_values(by='Weight_Magnitude', ascending=False)
    # feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # plot loss and accuracy curves
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    loss_csv_path = os.path.join(MODEL_DIR, "training_loss.csv")
    if os.path.exists(loss_csv_path):
        loss_df = pd.read_csv(loss_csv_path)
        train_losses = loss_df['train_loss'].tolist()
        val_losses = loss_df['val_loss'].tolist()
        train_accuracies = loss_df['train_accuracy'].tolist()
        val_accuracies = loss_df['val_accuracy'].tolist()

        axes[0].plot(train_losses, label='Training Loss')
        axes[0].plot(val_losses, label='Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Test Loss over Epochs')
        axes[0].legend()
        axes[1].plot(train_accuracies, label='Training Accuracy')
        axes[1].plot(val_accuracies, label='Test Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Test Accuracy over Epochs')
        axes[1].legend()
        plt.savefig(os.path.join(output_dir, "training_curves_reg.png"), dpi=300)

# Compute correlation between model probabilities and true value, vs BMI and true value
def compute_correlation(model, data_loader, bmi, device, output_dir):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(features)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)

    correlations_model = []
    correlations_bmi = []
    num_conditions = all_labels.shape[1]
    for i in range(num_conditions):
        corr_model = np.corrcoef(all_outputs[:, i], all_labels[:, i])[0, 1]
        corr_bmi = np.corrcoef(bmi, all_labels[:, i])[0, 1]
        correlations_model.append(corr_model)
        correlations_bmi.append(corr_bmi)

    correlation_df = pd.DataFrame({
        'Condition': data_loader.dataset.label_cols,
        'Correlation_Model': correlations_model,
        'Correlation_BMI': correlations_bmi
    })
    correlation_df.to_csv(os.path.join(output_dir, "correlation_analysis.csv"), index=False)

if __name__ == "__main__":
    DATA_DIR = r"data"
    MODEL_DIR = r"models"
    OUTPUT_DIR = r"analysis"
    CSV_FILE = os.path.join(DATA_DIR, "nhanes_merged_complete.csv")
    MODEL_LOAD_PATH = os.path.join(MODEL_DIR, "best_deep_neural_network_dnn.pth")
    LOSS_SAVE_PATH = os.path.join(MODEL_DIR, "training_loss.csv")
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    dataset = IBMIDataset(csv_file=CSV_FILE)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Load just BMI from csv
    bmi = pd.read_csv(CSV_FILE)['BMI'].values.astype('float32')

    # Load model
    model = DeepNeuralNetwork(input_dim=dataset.num_features, output_dim=dataset.num_labels)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    model.to(DEVICE)

    # Analyze model
    analyze_model(model, data_loader, DEVICE, OUTPUT_DIR)
    # compute_correlation(model, data_loader, bmi, DEVICE, OUTPUT_DIR)