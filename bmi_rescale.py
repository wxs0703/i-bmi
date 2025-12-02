import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from deep_neural_network import DeepNeuralNetwork, DeepResidualNetwork
from edited_bmi_data import IBMIDataset
import os

class BMILinearCombiner:
    def __init__(self, bmi_min=15.0, bmi_max=30.0, weights=None, num_outputs=5):
        self.bmi_min = bmi_min
        self.bmi_max = bmi_max
        self.bmi_range = bmi_max - bmi_min
        self.num_outputs = num_outputs
        
        if weights is None:
            self.weights = np.ones(num_outputs) / num_outputs
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum() 
    
    def combine_to_bmi(self, probabilities):

        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
    
        combined = np.dot(probabilities, self.weights)
        bmi_values = self.bmi_min + combined * self.bmi_range
        return bmi_values, combined
    
    def set_weights(self, weights):
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()


def predict_and_combine(model, data_loader, combiner, device='cpu'):
    model.eval()
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            features = batch['features'].to(device)
            outputs = model(features)  
            all_probabilities.append(outputs.cpu().numpy())
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
    
    probabilities = np.vstack(all_probabilities)
    num_outputs = probabilities.shape[1]

    bmi_predictions, combined_probs = combiner.combine_to_bmi(probabilities)

    results = {}
    for i in range(num_outputs):
        results[f'prob_{i+1}'] = probabilities[:, i]
    
    results['combined_prob'] = combined_probs
    results['bmi_prediction'] = bmi_predictions
    
    results_df = pd.DataFrame(results)
    return results_df, bmi_predictions

def evaluate_predictions(bmi_predictions, true_bmi_values):
    mae = np.mean(np.abs(bmi_predictions - true_bmi_values))
    mse = np.mean((bmi_predictions - true_bmi_values) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((bmi_predictions - true_bmi_values) / true_bmi_values)) * 100
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}

def optimize_weights(probabilities, true_bmi_values, bmi_min, bmi_max, num_outputs):
    from itertools import product
    best_mae = float('inf')
    best_weights = None
    weight_options = np.arange(0, 1.1, 0.2)
    count = 0
    for weights_tuple in product(weight_options, repeat=num_outputs-1):
        last_weight = 1 - sum(weights_tuple)
        if last_weight < 0 or last_weight > 1:
            continue
        
        weights = np.array(list(weights_tuple) + [last_weight])
        combined = np.dot(probabilities, weights)
        bmi_pred = bmi_min + combined * (bmi_max - bmi_min)
        
        mae = np.mean(np.abs(bmi_pred - true_bmi_values))
        
        if mae < best_mae:
            best_mae = mae
            best_weights = weights
        
        count += 1
    return best_weights

def main():
    MODEL_TYPE = 'dnn'
    DATA_DIR = r"data"
    MODEL_DIR = r"models"
    CSV_FILE = os.path.join(DATA_DIR, "nhanes_merged_complete.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, f"best_deep_neural_network_{MODEL_TYPE}.pth")
    OUTPUT_PATH = os.path.join(MODEL_DIR, f"bmi_predictions_{MODEL_TYPE}.csv")
    
    BMI_MIN = 15.0
    BMI_MAX = 50.0
    WEIGHTS = None  
    NUM_OUTPUTS = None
    OPTIMIZE_WEIGHTS = False
    
    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = IBMIDataset(csv_file=CSV_FILE)

    # Create data loader
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Rebuild model structure
    input_dim = dataset.num_features
    output_dim = dataset.num_labels
    
    if MODEL_TYPE == 'dnn':
        model = DeepNeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[128, 64, 32],
            dropout_rate=0.2
        )
    elif MODEL_TYPE == 'resnet':
        model = DeepResidualNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=128,
            num_blocks=3,
            dropout_rate=0.2
        )
    else:
        raise ValueError(f"MODEL_TYPE must be 'dnn' or 'resnet'")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    if NUM_OUTPUTS is None:
        num_outputs = output_dim
    else:
        num_outputs = NUM_OUTPUTS
    
    print(f'Number of model outputs: {num_outputs}')
    combiner = BMILinearCombiner(bmi_min=BMI_MIN, bmi_max=BMI_MAX, weights=WEIGHTS, num_outputs=num_outputs)
    print(f'BMI range: [{BMI_MIN}, {BMI_MAX}]\n')

    results_df, bmi_predictions = predict_and_combine(
        model, data_loader, combiner, device
    )
    
    if OPTIMIZE_WEIGHTS:
        try:
            model.eval()
            all_probs = []
            with torch.no_grad():
                for batch in data_loader:
                    features = batch['features'].to(device)
                    outputs = model(features)
                    all_probs.append(outputs.cpu().numpy())
            all_probs = np.vstack(all_probs)
            
        except Exception as e:
            print(f"Weight optimization failed: {e}")
    
   
    results_df.to_csv(OUTPUT_PATH, index=False)
        
    print("\n" + "="*80)
    print("Complete Data Table for First 10 Samples")
    print("="*80)
    print(results_df.head(10).to_string(index=True))
    

if __name__ == "__main__":
    main()