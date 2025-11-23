import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from data import IBMIDataset
from network import MultivariateLogisticRegression
from torch.utils.data import DataLoader

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_accuracy_curve(train_accuracies, val_accuracies, save_path):
    plt.figure()
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def analyze_model(model, data_loader, device, output_dir):
    
    # analyze model weights
    weights = model.linear.weight.data.cpu().numpy()
    num_conditions = weights.shape[0]
    feature_names = data_loader.dataset.features.columns.tolist()
    # take L2 norm of weights for each feature across all conditions, to get overall importance
    weight_magnitudes = np.linalg.norm(weights, axis=0)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Weight_Magnitude': weight_magnitudes
    }).sort_values(by='Weight_Magnitude', ascending=False)
    feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # plot loss and accuracy curves
    loss_csv_path = os.path.join(MODEL_DIR, "training_loss.csv")
    if os.path.exists(loss_csv_path):
        loss_df = pd.read_csv(loss_csv_path)
        train_losses = loss_df['train_loss'].tolist()
        val_losses = loss_df['val_loss'].tolist()
        train_accuracies = loss_df['train_accuracy'].tolist()
        val_accuracies = loss_df['val_accuracy'].tolist()

        plot_loss_curve(train_losses, val_losses, os.path.join(output_dir, "loss_curve.png"))
        plot_accuracy_curve(train_accuracies, val_accuracies, os.path.join(output_dir, "accuracy_curve.png"))


if __name__ == "__main__":
    DATA_DIR = r"data"
    MODEL_DIR = r"models"
    OUTPUT_DIR = r"analysis"
    CSV_FILE = os.path.join(DATA_DIR, "nhanes_merged_complete.csv")
    MODEL_LOAD_PATH = os.path.join(MODEL_DIR, "multivariate_logistic_regression.pth")
    LOSS_SAVE_PATH = os.path.join(MODEL_DIR, "training_loss.csv")
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    dataset = IBMIDataset(csv_file=CSV_FILE)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = MultivariateLogisticRegression(input_dim=dataset.num_features, output_dim=dataset.num_labels)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    model.to(DEVICE)

    # Analyze model
    analyze_model(model, data_loader, DEVICE, OUTPUT_DIR)