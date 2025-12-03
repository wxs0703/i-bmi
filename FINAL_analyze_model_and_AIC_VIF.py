# Importing packages and functions
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from bmi_data import IBMIDataset
from network import MultivariateLogisticRegression
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# A class for the feature dataset used in the AIC runs
class AICFeatureDataset(Dataset):
  
    # Wrap IBMIDataset but restrict to a subset of feature columns.
   
    def __init__(self, base_dataset, feature_names):
        self.base = base_dataset
        self.feature_names = list(feature_names)

        full_feature_df = self.base.features
        label_df = self.base.labels

        self.X = full_feature_df[self.feature_names].to_numpy(dtype="float32")
        self.y = label_df.to_numpy(dtype="float32")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y

# Function used to train temporary reduced models in the AIC analysis
def train_temp_model(dataset, device, num_epochs=10, lr=1e-3, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.X.shape[1]
    output_dim = dataset.y.shape[1]

    # Trains with the same multivariate logistic regression (MLR) model that was
    # originally used
    model = MultivariateLogisticRegression(input_dim=input_dim,
                                           output_dim=output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        n_samples = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            probs = model(X_batch)
            loss = criterion(probs, y_batch)
            loss.backward()
            optimizer.step()

            bs = X_batch.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

        print(f"[Temp] Epoch {epoch+1}/{num_epochs} - loss: {running_loss / n_samples:.4f}")

    return model

# Function to plot the loss curve function from the MLR
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

# Function to plot the accuracy curve from the MLR
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

# Function to analyze the trained MLR model by looking at: model weights, L2 norm of weights, and plots loss and
# accuracy curves
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

# Function to compute the log likelihood utilized in the AIC calculation
def compute_log_likelihood(model, data_loader, device):
    model.eval()
    total_log_likelihood = 0.0
    bce_sum = nn.BCELoss(reduction="sum")  

    with torch.no_grad():
        for batch in data_loader:

            if isinstance(batch, dict):
                X_batch = batch["features"].to(device)
                y_batch = batch["labels"].to(device).float()
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).float()

            probs = model(X_batch)          

            # BCE returns negative log-likelihood per sample summed
            nll_batch = bce_sum(probs, y_batch) 
            ll_batch = -nll_batch                

            if torch.isnan(ll_batch) or torch.isinf(ll_batch):
                return float("nan")

            total_log_likelihood += ll_batch.item()

    return total_log_likelihood

# Function to count the total number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to perfom the AIC calculation of 2k - 2(log likelihood)
def compute_aic(model, data_loader, device):
    log_likelihood = compute_log_likelihood(model, data_loader, device)
    k = count_parameters(model)
    aic = 2 * k - 2 * log_likelihood
    return aic

# Runs AIC through backward selection where we start with a full model and then iteratively remove a feature
# until we get a model that balances model complexity and fit to the data (looking for lowest AIC)
def backward_aic_feature_selection(base_dataset, device, num_epochs=10, lr=1e-3, batch_size=256, min_improvement=0.0):
    all_features = base_dataset.features.columns.tolist()
    print(f"Backward AIC selection on {len(all_features)} candidate features.")

    current_features = all_features.copy()

    # Baseline model with all features
    full_dataset = AICFeatureDataset(base_dataset, current_features)
    full_model = train_temp_model(full_dataset, device,
                                  num_epochs=num_epochs,
                                  lr=lr,
                                  batch_size=batch_size)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    # Computes the initial AIC of the full model
    best_aic = compute_aic(full_model, full_loader, device)
    print(f"Initial AIC with all features ({len(current_features)}): {best_aic}")

    # Loop that performs the backward AIC selection
    improved = True
    while improved and len(current_features) > 1:
        improved = False
        best_candidate_aic = best_aic
        feature_to_drop = None

        # Loops through each feature and removes one, and calulates AIC of a trained model without that feature
        for feat in current_features:
            trial_features = [f for f in current_features if f != feat]
            trial_dataset = AICFeatureDataset(base_dataset, trial_features)
            trial_model = train_temp_model(trial_dataset, device,
                                           num_epochs=num_epochs,
                                           lr=lr,
                                           batch_size=batch_size)
            trial_loader = DataLoader(trial_dataset, batch_size=batch_size, shuffle=False)
            trial_aic = compute_aic(trial_model, trial_loader, device)

            print(f"Dropping {feat:30s} -> AIC = {trial_aic}")

            # Edge case if the log likelihood of AIC creates a NAN or inf value
            if (trial_aic is None or np.isnan(trial_aic) or np.isinf(trial_aic)):
                continue

            # Comparison where if an AIC from a model without a feature is lower than the current best AIC
            # it assigns the lower AIC as the best AIC
            if trial_aic + min_improvement < best_candidate_aic:
                best_candidate_aic = trial_aic
                feature_to_drop = feat

        # Report back the feature we dropped
        if feature_to_drop is not None:
            current_features.remove(feature_to_drop)
            best_aic = best_candidate_aic
            improved = True
            print(f"DROPPED {feature_to_drop} | new best AIC = {best_aic}")
            print(f"Remaining features: {len(current_features)}")
        else:
            print("No further AIC improvement by dropping a single feature.")

    # Final message that says what is the total amount of final features we have + its AIC
    # also saves a CSV file later down in main of which features were saves
    print("\n=== BACKWARD AIC SELECTION COMPLETE ===")
    print(f"Selected {len(current_features)} features with AIC = {best_aic}")
    return current_features, best_aic

if __name__ == "__main__":
    DATA_DIR = "your path to data directory"
    MODEL_DIR = "your path to model directory"
    OUTPUT_DIR = "your path to output directory"
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
    
    # Backward AIC feature selection 
    selected_features, best_aic = backward_aic_feature_selection(base_dataset=dataset, device=DEVICE, num_epochs=50, lr=1e-3, batch_size=256, min_improvement=0.0)

    # Save selected features to a CSV for later use in training
    selected_path = os.path.join(OUTPUT_DIR, "aic_selected_features.csv")
    pd.DataFrame({"feature": selected_features}).to_csv(selected_path, index=False)
    print(f"Saved selected feature list to {selected_path}")

    # Correlation matrix between all features
    corr = dataset.features.corr()   

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.show()

    # Reports VIF of each feature
    X = dataset.features.copy()
    X = X.fillna(0) 

    vif_df = pd.DataFrame()
    vif_df["Feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif_df.sort_values(by="VIF", ascending=False))



