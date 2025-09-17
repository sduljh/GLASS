import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv
import time
import torch
import torch.nn as nn
from dgl.nn import GraphConv
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import dgl
import numpy as np
import random
import pandas as pd
import os
from align_compute import train_preprocess, test_preprocess
from anno_process import process_junction
from graph_construction import cons_graph
import logging
import torch
import dgl
import numpy as np
import pandas as pd


class BipartiteGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BipartiteGCN, self).__init__()
        self.convs = nn.ModuleList([
            GraphConv(in_channels, 64),
            GraphConv(64, 128),
            GraphConv(128, 128),
            GraphConv(128, 128),
            GraphConv(128, 128),
            GraphConv(128, out_channels)
        ])
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(6)])
        self.att_weights = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(6)])
        self.att_activations = nn.ModuleList([nn.Sigmoid() for _ in range(6)])
        self.fc = nn.Linear(out_channels, 2)

    def forward(self, graph):
        # Initialize intron node features with random values
        graph.nodes['intron'].data['feat'] = torch.randn(
            graph.number_of_nodes('intron'), 10, device=graph.device
        )
        h_read = graph.nodes['read'].data['feat']
        h_intron = graph.nodes['intron'].data['feat']

        # Alternate GCN propagation between read and intron nodes
        for i in range(6):
            if i % 2 == 0:
                # Propagate from read nodes to intron nodes
                edge_index = graph['read', 'to_intron', 'intron']
                h_intron = self.convs[i](edge_index, h_read)
                # Apply attention-weighted activation
                h_intron = h_intron * self.att_activations[i](self.att_weights[i])
                h_intron = torch.relu(h_intron)
            else:
                # Propagate from intron nodes back to read nodes
                edge_index = graph['intron', 'to_read', 'read']
                h_read = self.convs[i](edge_index, h_intron)
                # Apply attention-weighted activation
                h_read = h_read * self.att_activations[i](self.att_weights[i])
                h_read = torch.relu(h_read)

        # Final classification layer
        out = self.fc(h_read)
        return out


def compute_feature_importance(model, graph, device):
    """Compute feature importance using gradient-based method"""
    # Clone read node features and enable gradient calculation
    feat = graph.nodes['read'].data['feat'].detach().clone()
    feat.requires_grad_(True)
    graph.nodes['read'].data['feat'] = feat

    # Forward pass
    model.eval()
    output = model(graph)

    # Get prediction probabilities
    probs = torch.softmax(output, dim=1)
    max_probs = probs.max(dim=1).values

    # Compute gradients
    model.zero_grad()
    max_probs.sum().backward()

    # Get absolute feature gradients
    gradients = torch.abs(feat.grad).cpu().numpy()

    # Feature importance = average gradient across all nodes
    feature_importance = gradients.mean(axis=0)
    return feature_importance


def evaluate_feature_importance():
    # Set device configuration
    device = torch.device('cpu')

    # Load validation graph
    val_graph, _ = dgl.load_graphs('./val_graph.dgl')
    val_graph = val_graph[0].to(device)

    # Load model ensemble and F1 scores
    models = torch.load('./trained_models.pth')
    model_f1_scores = torch.load('./GLASS_code/model_f1_scores.pth')
    weights = [f1 / sum(model_f1_scores) for f1 in model_f1_scores]

    # Compute feature importance for each model
    all_importances = []
    for model, weight in zip(models, weights):
        model.to(device)
        importance = compute_feature_importance(model, val_graph, device)
        all_importances.append(importance * weight)  # Weighted importance

    # Aggregate importance across all models
    avg_importance = np.sum(all_importances, axis=0)

    # Create importance DataFrame
    results = []
    num_features = val_graph.nodes['read'].data['feat'].shape[1]
    for idx in range(num_features):
        results.append({
            'feature_index': idx,
            'importance': avg_importance[idx],
            'normalized_importance': avg_importance[idx] / avg_importance.sum()
        })

    # Sort by descending importance
    results.sort(key=lambda x: x['importance'], reverse=True)
    importance_df = pd.DataFrame(results)

    # Save results
    importance_df.to_csv('./feature_importance.csv', index=False)
    print("Feature importance results saved")

    # Display top 5 features
    print("\n===== Top 5 Important Features =====")
    print(importance_df.head())
    return importance_df


if __name__ == "__main__":
    importance_df = evaluate_feature_importance()