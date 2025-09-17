import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import dgl
import numpy as np
import random
import pandas as pd
import time


# Set up device
device = torch.device('cuda')
print(f'Device: {device}')


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True  # ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # disable non-deterministic algorithms


# Define a simple neural network model for classification
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second layer
        # self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        return x


# Label smoothing loss function
def label_smoothed_nll_loss(logits, target, eps=2):
    num_classes = logits.size(1)
    one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
    smoothed_labels = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    loss = -(smoothed_labels * log_probs).sum(dim=1).mean()
    return loss


# Training function
def train_model(model, train_graph, train_labels, val_graph, val_labels, optimizer, epochs=100, eps=0.2, fruit='fruit'):
    best_val_f1 = -1  # Track the best validation F1 score
    best_epoch = 0    # Track the epoch of best performance
    best_model = None  # Store the best model
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Extract read node features
        read_features = train_graph.nodes['read'].data['feat']

        # Forward pass
        out = model(read_features)
        loss = label_smoothed_nll_loss(out, train_labels, eps=eps)

        loss.backward()
        optimizer.step()

        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_read_features = val_graph.nodes['read'].data['feat']
                val_out = model(val_read_features)

                val_preds = torch.argmax(val_out, dim=1)
                val_loss = label_smoothed_nll_loss(val_out, val_labels, eps=eps)

                # Calculate weighted F1 score
                f1 = f1_score(val_labels.cpu(), val_preds.cpu(), average='weighted')

                if epoch > 1 and f1 < 0.999999 and f1 > best_val_f1:
                    best_val_f1 = f1
                    best_epoch = epoch
                    best_model = model
                    torch.save(best_model, 'best_model.pth')

            model.train()
    print(f'Best F1 Score: {best_val_f1:.4f} at epoch {best_epoch}')
    return best_model, best_val_f1


# Testing function
def test_model(models, weights, graph, labels, output_file):
    for model in models:
        model.to(device)
        model.eval()

    graph = graph.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        all_probs = []
        for model in models:
            read_features = graph.nodes['read'].data['feat']
            out = model(read_features)
            probs = torch.nn.functional.softmax(out, dim=1)
            all_probs.append(probs)

        weighted_probs = sum(w * p for w, p in zip(weights, all_probs))

        predictions = torch.argmax(weighted_probs, dim=1)

        # Calculate metrics
        accuracy = (predictions == labels).float().mean()
        f1 = f1_score(labels.cpu(), predictions.cpu(), average='weighted')
        conf_matrix = confusion_matrix(labels.cpu(), predictions.cpu())
        precision = precision_score(labels.cpu(), predictions.cpu(), average=None)
        recall = recall_score(labels.cpu(), predictions.cpu(), average=None)

        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        for i, (p, r) in enumerate(zip(precision, recall)):
            print(f'Class {i} - Precision: {p:.4f}, Recall: {r:.4f}')

        # Save predictions
        pred_labels = predictions.to(graph.device)
        read_id2s = graph.nodes['read'].data['read_id2']

        df = pd.DataFrame({
            'read_id2': read_id2s.cpu().numpy(),
            'Predicted_Label': pred_labels.cpu().numpy()
        })

        df.to_csv(output_file, sep='\t', index=False)
        print(f"Predictions saved to {output_file}")


# Main function
def main():
    start_time = time.time()
    set_seed(42)

    train_fruits = ["SRR25180684", "ERR4706159", "SRR23347361", "ERR9286493", "DRR228524"]
    test_fruits = ["SRR23513619"]
    models = []
    model_f1_scores = []

    # Load validation data
    val_fruit = 'SRR1163657'
    val_graph, _ = dgl.load_graphs(f'./{val_fruit}.dgl')
    val_graph = val_graph[0].to(device)
    val_labels = val_graph.nodes['read'].data['label'].to(device)

    # Train models
    for fruit in train_fruits:
        train_graph, _ = dgl.load_graphs(f'./{fruit}.dgl')
        train_graph = train_graph[0].to(device)
        train_labels = train_graph.nodes['read'].data['label'].to(device)

        # Model configuration
        in_channels = train_graph.nodes['read'].data['feat'].shape[1]
        hidden_dim = 128
        output_dim = 2  # Binary classification
        epochs = 200
        model = SimpleNN(in_channels, hidden_dim, output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        print(f"Training model for {fruit}...")
        best_model, best_val_f1 = train_model(model, train_graph, train_labels, val_graph, val_labels, optimizer,
                                              epochs=epochs, eps=0.1, fruit=fruit)

        models.append(best_model)
        model_f1_scores.append(best_val_f1)

    # Calculate ensemble weights
    total_f1 = sum(model_f1_scores)
    weights = [f1 / total_f1 for f1 in model_f1_scores]

    # Test models
    for fruit in test_fruits:
        test_graph, _ = dgl.load_graphs(f'./{fruit}.dgl')
        test_graph = test_graph[0]
        test_labels = test_graph.nodes['read'].data['label']

        print(f"Testing on {fruit}...")
        output_file = f"./{fruit}_predictions.csv"
        test_model(models, weights, test_graph, test_labels, output_file)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
