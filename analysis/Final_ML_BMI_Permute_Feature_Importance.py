# Importing packages 
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Function to evalute model based on its accuracy
def evaluate_model(model, features, labels, device, batch_size = 128):
    model.eval()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_correct = 0.0
    total_examples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)     
            probs = outputs         

            preds = (probs >= 0.5).float()
            
            # multi-label accuracy: mean over labels, then mean over batch
            batch_acc = (preds == y).float().mean(dim=1)  
            total_correct += batch_acc.sum().item()
            total_examples += x.size(0)

    avg_acc = total_correct / total_examples
    return avg_acc
    

# Function to perform permutation feature importance
def permutation_importance(model, dataset, device, batch_size=128, n_repeats=5):
  
    # Get feature names from IBMIDataset
    feature_names = dataset.features.columns.tolist()

    # Collect all data into tensors
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    all_labels = []
    for batch in loader:
        all_features.append(batch["features"])
        all_labels.append(batch["labels"])

    features = torch.cat(all_features, dim=0).to(device)   
    labels = torch.cat(all_labels, dim=0).to(device)    

    N, D = features.shape

    assert len(feature_names) == D, f"Got {len(feature_names)} names for {D} features"

    # Baseline accuracy
    baseline_acc = evaluate_model(model, features, labels, device, batch_size=batch_size)

    results = []

    # Loop over features and calculates the feature importance of the model after permuting feature
    for j in range(D):
        accs = []
        for _ in range(n_repeats):
            perm_features = features.clone()
            idx = torch.randperm(N)
            perm_features[:, j] = perm_features[idx, j]

            perm_acc = evaluate_model(model, perm_features, labels, device, batch_size=batch_size)
            accs.append(perm_acc)

        mean_perm_acc = float(sum(accs) / len(accs))

        # Calculates a difference in accuracy from the permuted model compared to the baseline, original model
        delta_acc = mean_perm_acc - float(baseline_acc)

        results.append({
            "feature_index": j,
            "feature_name": feature_names[j],
            "baseline_accuracy": float(baseline_acc),
            "permuted_accuracy": mean_perm_acc,
            "delta_accuracy": delta_acc,
        })

    results_sorted = sorted(results, key=lambda d: d["delta_accuracy"])
    df = pd.DataFrame(results_sorted)
    return df
