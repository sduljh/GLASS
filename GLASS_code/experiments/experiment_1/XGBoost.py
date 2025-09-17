import torch
import xgboost as xgb  # Import XGBoost
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import dgl
import numpy as np
import random
import pandas as pd
import time

# Set up device
device = torch.device('cpu')
print(f'Device: {device}')


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable non-deterministic algorithms


# Define XGBoost model training function (binary classification)
def train_xgboost(train_graph, train_labels):
    # Extract read node features from training data
    features = train_graph.nodes['read'].data['feat'].cpu().numpy()
    labels = train_labels.cpu().numpy()

    # Convert data to DMatrix format required by XGBoost
    dtrain = xgb.DMatrix(features, label=labels)

    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic',  # Binary classification with probability output
        'eval_metric': 'logloss',        # Use logloss as evaluation metric
        'max_depth': 6,
        'learning_rate': 0.1,
        'nthread': 4
    }

    # Train XGBoost model
    bst = xgb.train(params, dtrain, num_boost_round=100)

    return bst


# Define model testing function
def test_model(models, weights, graph, labels, output_file):
    graph = graph.to(device)
    labels = labels.to(device)

    # XGBoost models are CPU-based, no need for device conversion
    with torch.no_grad():
        # Aggregate probabilities from all models
        all_probs = []
        for model in models:
            # For XGBoost models, use predict_proba to get probabilities
            features = graph.nodes['read'].data['feat'].cpu().numpy()
            dtest = xgb.DMatrix(features)
            probs = model.predict(dtest)  # Get probabilities
            all_probs.append(torch.tensor(probs, device=graph.device))

        # Calculate weighted average probabilities
        weighted_probs = sum(w * p for w, p in zip(weights, all_probs))

        # Get final predictions (using 0.5 as threshold)
        predictions = (weighted_probs > 0.5).long()

        # Calculate evaluation metrics
        accuracy = (predictions == labels).float().mean()
        f1 = f1_score(labels.cpu(), predictions.cpu(), average='weighted')
        conf_matrix = confusion_matrix(labels.cpu(), predictions.cpu())
        precision = precision_score(labels.cpu(), predictions.cpu(), average=None)
        recall = recall_score(labels.cpu(), predictions.cpu(), average=None)

        # Print results
        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        for i, (p, r) in enumerate(zip(precision, recall)):
            print(f'Class {i} - Precision: {p:.4f}, Recall: {r:.4f}')

        # Save predictions to graph node attributes
        pred_labels = predictions.to(graph.device)
        read_id2s = graph.nodes['read'].data['read_id2']

        # Create DataFrame with predictions
        df = pd.DataFrame({
            'read_id2': read_id2s.cpu().numpy(),
            'Predicted_Label': pred_labels.cpu().numpy()
        })

        # Save to CSV file
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Predictions have been saved to {output_file}")


# Main function
def main():
    start_time = time.time()  # Record start time
    set_seed(42)

    # Load training data
    train_fruits = ["SRR25180684", "ERR4706159", "SRR23347361", "DRR228524", "ERR9286493"]
    test_fruits = ["SRR23513619"]
    models = []
    model_f1_scores = []

    # Load validation data
    val_fruit = 'SRR1163657'
    val_graph, _ = dgl.load_graphs(f'./{val_fruit}.dgl')
    val_graph = val_graph[0].to(device)
    val_labels = val_graph.nodes['read'].data['label'].to(device)

    # Train multiple classifiers
    for fruit in train_fruits:
        # Load training data
        train_graph, _ = dgl.load_graphs(f'./{fruit}.dgl')
        train_graph = train_graph[0].to(device)
        train_labels = train_graph.nodes['read'].data['label'].to(device)

        # Train XGBoost model
        print(f"Training model for {fruit}...")
        model = train_xgboost(train_graph, train_labels)

        # Save model and calculate F1 score
        models.append(model)
        model_f1_scores.append(f1_score(train_labels.cpu(),
            (model.predict(xgb.DMatrix(train_graph.nodes['read'].data['feat'].cpu().numpy())) > 0.5).astype(int),
            average='weighted'))

    # Calculate ensemble weights
    total_f1 = sum(model_f1_scores)
    weights = [f1 / total_f1 for f1 in model_f1_scores]

    # Test models
    for fruit in test_fruits:
        test_graph, _ = dgl.load_graphs(f'./{fruit}.dgl')
        test_graph = test_graph[0]
        test_labels = test_graph.nodes['read'].data['label']

        # Run testing
        print(f"Testing on {fruit}...")
        output_file = f"./{fruit}_predictions.csv"
        test_model(models, weights, test_graph, test_labels, output_file)

    end_time = time.time()  # Record end time
    print(f"Execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
