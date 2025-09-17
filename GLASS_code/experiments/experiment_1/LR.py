import torch
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression classifier
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
    torch.backends.cudnn.benchmark = False  # Prevent GPU from using non-deterministic algorithms


# Define logistic regression model training function
def train_logistic_regression(train_graph, train_labels):
    # Extract features from the training graph
    features = train_graph.nodes['read'].data['feat'].cpu().numpy()
    labels = train_labels.cpu().numpy()

    # Train the logistic regression classifier
    clf = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')  # Use liblinear solver
    clf.fit(features, labels)

    return clf


# Define model testing function
def test_model(models, weights, graph, labels, output_file):
    graph = graph.to(device)
    labels = labels.to(device)

    # Logistic regression model is CPU-based, so no need to call to() method
    with torch.no_grad():
        # Collect the probabilities from all models
        all_probs = []
        for model in models:
            # For logistic regression, use the predict_proba method to get probabilities
            features = graph.nodes['read'].data['feat'].cpu().numpy()
            probs = model.predict_proba(features)  # Get probabilities
            all_probs.append(torch.tensor(probs, device=graph.device))

        # Weighted average of all model probabilities
        weighted_probs = sum(w * p for w, p in zip(weights, all_probs))

        # Get final labels based on weighted probabilities
        predictions = torch.argmax(weighted_probs, dim=1)

        # Compute evaluation metrics
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

        # Save predictions as node attributes in the graph
        pred_labels = predictions.to(graph.device)  # Ensure predicted labels are on the correct device
        read_id2s = graph.nodes['read'].data['read_id2']

        # Get the indices of the nodes (based on node type)
        read_id2s = read_id2s.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()

        # Create a DataFrame to store the predicted labels
        df = pd.DataFrame({
            'read_id2': read_id2s,
            'Predicted_Label': pred_labels
        })

        # Save to a CSV file
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Predictions have been saved to {output_file}")


# Main function
def main():
    start_time = time.time()  # Record the start time
    set_seed(42)

    # Load training data
    train_fruits = ["SRR25180684", "ERR4706159", "SRR23347361", "DRR228524", "ERR9286493"]
    # train_fruits = ["SRR25180684", "ERR4706159"]

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

        # Train the logistic regression model
        print(f"Training model for {fruit}...")
        model = train_logistic_regression(train_graph, train_labels)

        # Save the trained model and its best f1_score
        models.append(model)
        model_f1_scores.append(
            f1_score(train_labels.cpu(), model.predict(train_graph.nodes['read'].data['feat'].cpu().numpy()),
                     average='weighted'))

    total_f1 = sum(model_f1_scores)
    weights = [f1 / total_f1 for f1 in model_f1_scores]

    # Test the model
    for fruit in test_fruits:
        test_graph, _ = dgl.load_graphs(f'./{fruit}.dgl')
        test_graph = test_graph[0]
        test_labels = test_graph.nodes['read'].data['label']

        print(f"Testing on {fruit}...")
        output_file = f"./{fruit}_predictions.csv"
        test_model(models, weights, test_graph, test_labels, output_file)

    end_time = time.time()  # Record the end time
    print(f"Total runtime: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()

