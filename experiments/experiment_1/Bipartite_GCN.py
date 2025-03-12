import torch
import torch.nn as nn
from dgl.nn import GraphConv
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import dgl
import numpy as np
import random
import pandas as pd
import time

# Set devices
train_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_device = torch.device('cpu')
print(f'Training on device: {train_device}')
print(f'Testing on device: {test_device}')


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable non-deterministic algorithms


# Label smoothing loss function
def label_smoothed_nll_loss(logits, target, eps=2):
    num_classes = logits.size(1)
    one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
    smoothed_labels = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    loss = -(smoothed_labels * log_probs).sum(dim=1).mean()
    return loss


# Define Bipartite Graph Convolutional Network model
class BipartiteGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BipartiteGCN, self).__init__()

        # Define graph convolutional layers
        self.convs = nn.ModuleList([
            GraphConv(in_channels, 64),
            GraphConv(64, 128),
            GraphConv(128, 128),
            GraphConv(128, 128),
            GraphConv(128, 128),
            GraphConv(128, out_channels)
        ])

        # Initialize parameters
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(6)
        ])
        self.att_weights = nn.ParameterList([
            nn.Parameter(torch.randn(1)) for _ in range(6)
        ])
        self.att_activations = nn.ModuleList([
            nn.Sigmoid() for _ in range(6)
        ])

        self.fc = nn.Linear(out_channels, 2)

    def forward(self, graph):
        # Initialize intron features using graph connections
        graph.nodes['intron'].data['feat'] = torch.randn(
            graph.number_of_nodes('intron'), 10, device=graph.device
        )

        # Update features using graph convolutional layers and activation functions
        h_read = graph.nodes['read'].data['feat']
        h_intron = graph.nodes['intron'].data['feat']

        for i in range(6):
            if i % 2 == 0:
                edge_index = graph['read', 'to_intron', 'intron']
                h_intron = self.convs[i](edge_index, h_read)
                h_intron = h_intron * self.att_activations[i](self.att_weights[i])
                h_intron = torch.relu(h_intron)
            else:
                edge_index = graph['intron', 'to_read', 'read']
                h_read = self.convs[i](edge_index, h_intron)
                h_read = h_read * self.att_activations[i](self.att_weights[i])
                h_read = torch.relu(h_read)

        out = self.fc(h_read)
        return out


# Model training function
def train_model(model, train_graph, train_labels, val_graph, val_labels, optimizer, epochs=100, eps=0.2, fruit='fruit'):
    best_val_f1 = -1  # Track best validation F1 score
    best_epoch = 0  # Track epoch of best performance
    best_model = None  # Store best model
    val_losses = []  # Track validation losses
    train_losses = []  # Track training losses

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass for training data
        out = model(train_graph)
        loss = label_smoothed_nll_loss(out, train_labels, eps=eps)
        loss.backward()
        optimizer.step()

        # Record training loss
        train_losses.append(loss.item())

        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Get validation predictions and compute metrics
                val_out = model(val_graph)
                val_preds = torch.argmax(val_out, dim=1)
                val_loss = label_smoothed_nll_loss(val_out, val_labels, eps=eps)
                val_losses.append(val_loss.item())

                # Calculate F1 score (weighted average)
                f1 = f1_score(val_labels.cpu(), val_preds.cpu(), average='weighted')

                if epoch > 1 and f1 < 0.999999 and f1 > best_val_f1:
                    best_val_f1 = f1
                    best_epoch = epoch
                    best_model = model
                    torch.save(best_model, 'best_model.pth')  # Save best model

            model.train()

    print(f'Best F1 Score: {best_val_f1:.4f} at epoch {best_epoch}')
    return best_model, best_val_f1


# Model testing function
def test_model(models, weights, graph, labels, output_file):
    # Ensure all models are in eval mode and moved to CPU
    for model in models:
        model.to(test_device)
        model.eval()

    graph = graph.to(test_device)
    labels = labels.to(test_device)

    with torch.no_grad():
        # Aggregate output probabilities from all models
        all_probs = []
        for model in models:
            out = model(graph)
            probs = torch.nn.functional.softmax(out, dim=1)
            all_probs.append(probs)

        # Weighted average of probabilities
        weighted_probs = sum(w * p for w, p in zip(weights, all_probs))

        # Get final predictions
        predictions = torch.argmax(weighted_probs, dim=1)

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

        # Add predicted labels to graph node attributes
        pred_labels = predictions.to(graph.device)
        read_id2s = graph.nodes['read'].data['read_id2']

        # Create prediction DataFrame
        df = pd.DataFrame({
            'read_id2': read_id2s.cpu().numpy(),
            'Predicted_Label': predictions.cpu().numpy()
        })

        # Save predictions
        print(f"Predictions have been saved to {output_file}")


# Main function
def main():
    start_time = time.time()  # Record start time
    set_seed(42)

    # Load training data
    train_fruits = ["SRR25180684", "ERR4706159", "SRR23347361", "DRR228524", "ERR9286493"]
    test_fruits = ["NA12878"]
    models = []
    model_f1_scores = []

    # Load validation data
    val_fruit = 'SRR1163657'
    val_graph, _ = dgl.load_graphs(f'./{val_fruit}.dgl')
    val_graph = val_graph[0].to(train_device)
    val_labels = val_graph.nodes['read'].data['label'].to(train_device)

    # Train multiple classifiers
    for fruit in train_fruits:
        # Load training graph
        train_graph, _ = dgl.load_graphs(f'./{fruit}.dgl')
        train_graph = train_graph[0].to(train_device)
        train_labels = train_graph.nodes['read'].data['label'].to(train_device)

        # Model configuration
        in_channels = train_graph.nodes['read'].data['feat'].shape[1]
        out_channels = 128
        epochs = 200
        model = BipartiteGCN(in_channels, out_channels).to(train_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        # Training process
        print(f"Training model for {fruit}...")
        best_model, best_val_f1 = train_model(model, train_graph, train_labels, val_graph, val_labels,
                                              optimizer, epochs=epochs, eps=0.1, fruit=fruit)

        models.append(best_model)
        model_f1_scores.append(best_val_f1)

    # Calculate ensemble weights
    total_f1 = sum(model_f1_scores)
    weights = [f1 / total_f1 for f1 in model_f1_scores]

    # Testing process
    for fruit in test_fruits:
        test_graph, _ = dgl.load_graphs(f'./{fruit}.dgl')
        test_graph = test_graph[0]
        test_labels = test_graph.nodes['read'].data['label']

        print(f"Testing on {fruit}...")
        output_file = f"./{fruit}_test_predictions.csv"
        test_model(models, weights, test_graph, test_labels, output_file)

    end_time = time.time()  # Record end time
    print(f"Execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()