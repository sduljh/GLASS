import torch
import torch.nn as nn
from dgl.nn import GraphConv
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import dgl
import numpy as np
import random
import pandas as pd
import os
import time

# Set device
train_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {train_device}')


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
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


# Define BipartiteGCN model
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

        # Update features using GCN layers and activations
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


# Training function
def get_loss(model, train_graph, train_labels, val_graph, val_labels, optimizer, epochs=100, eps=0.2, fruit='fruit'):
    best_val_f1 = -1  # Track best validation F1 score
    best_epoch = 0  # Track epoch of best performance
    best_model = None  # Store best model
    val_losses = []  # Track validation losses
    train_losses = []  # Track training losses

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
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

            model.train()

        # Save losses to CSV file
        if epoch % 10 == 0:
            data = {
                'Epoch': [epoch],
                'Train_Loss': [train_losses[-1]],
                'Val_Loss': [val_losses[-1]]
            }
            df = pd.DataFrame(data)
            df.to_csv(f'/home/lijiahao/refine/{fruit}_losses.csv',
                      mode='a',
                      header=not os.path.exists(f'/home/lijiahao/refine/{fruit}_losses.csv'),
                      index=False)

    print(f'Best F1 Score: {best_val_f1:.4f} at epoch {best_epoch}')
    return best_model, best_val_f1


# Main function
def main():
    start_time = time.time()  # Record start time
    set_seed(42)

    # Load training data
    train_fruits = ["SRR25180684", "ERR4706159", "SRR23347361", "DRR228524", "ERR9286493"]
    models = []
    model_f1_scores = []

    # Load validation data
    val_fruit = 'SRR1163657'
    val_graph, _ = dgl.load_graphs(f'../experiment_1/{val_fruit}.dgl')
    val_graph = val_graph[0].to(train_device)
    val_labels = val_graph.nodes['read'].data['label'].to(train_device)

    # Train multiple classifiers
    for fruit in train_fruits:
        # Load training graph
        train_graph, _ = dgl.load_graphs(f'../experiment_1/{fruit}.dgl')
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
        best_model, best_val_f1 = get_loss(model, train_graph, train_labels, val_graph, val_labels, optimizer,
                                           epochs=epochs, eps=0.1, fruit=fruit)

        models.append(best_model)
        model_f1_scores.append(best_val_f1)

    end_time = time.time()  # Record end time
    print(f"Execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
