import torch
import torch.nn as nn
from dgl.nn import GraphConv
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import dgl
import numpy as np
import random
import pandas as pd
import os
from data_preprocess import train_preprocess, test_preprocess, process_junction
from graph_construction import train_consgraph
from graph_construction import test_consgraph
import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True  # to ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # prevent GPU from using non-deterministic algorithms

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
        graph.nodes['intron'].data['feat'] = torch.randn(
            graph.number_of_nodes('intron'), 10, device=graph.device
        )
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

# Train model function
def train_model(model, train_graph, train_labels, val_graph, val_labels, optimizer, epochs=100, eps=0.2):
    best_val_f1 = -1  # Track the best precision on validation set
    best_epoch = 0  # Track the epoch of the best precision
    best_model = None  # To store the best model
    val_losses = []  # Used to record the loss value of the validation set
    train_losses = []  # Used to record the loss value of the train set
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass for training data
        out = model(train_graph)
        loss = label_smoothed_nll_loss(out, train_labels, eps=eps)

        loss.backward()
        optimizer.step()

        # Record the training loss
        train_losses.append(loss.item())
        # Evaluate on validation set every 10 epochs

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(val_graph)
                val_preds = torch.argmax(val_out, dim=1)
                val_loss = label_smoothed_nll_loss(val_out, val_labels, eps=eps)
                val_losses.append(val_loss.item())
                f1 = f1_score(val_labels.cpu(), val_preds.cpu(), average='weighted')

                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_epoch = epoch
                    best_model = model

            model.train()

    logger.info(f'Best F1 Score: {best_val_f1:.4f} at epoch {best_epoch}')


    return best_model, best_val_f1

# Main function: Train multiple models and save them
def data_train(train_junction_gtf_file = None, train_bam_files = None, val_bam_file = None, val_junction_gtf_file = None, train_device ='cpu'):
    set_seed(42)

    if val_bam_file is None or val_junction_gtf_file is None:
        val_graph, _ = dgl.load_graphs(f'./val_graph.dgl')
        val_graph = val_graph[0].to(train_device)
        val_labels = val_graph.nodes['read'].data['label'].to(train_device)
    else:
        val_annotation_path = os.path.dirname(val_junction_gtf_file)
        val_junction_input_file = os.path.join(val_annotation_path,
                                           f"{os.path.splitext(os.path.basename(val_junction_gtf_file))[0]}.info")
        junction_df = process_junction(val_junction_gtf_file, val_junction_input_file)


        val_bam_path = os.path.dirname(val_bam_file)
        # Add the suffix "_align" to the alignment.info file
        alignment_align_path = os.path.join(val_bam_path,
                                            f"{os.path.splitext(os.path.basename(val_bam_file))[0]}.info")
        # Call function
        feature_df, intron_df = train_preprocess(val_bam_file, alignment_align_path, junction_df)
        val_graph = train_consgraph(feature_df, intron_df, device = train_device)
        val_graph = val_graph[0].to(train_device)
        val_labels = val_graph.nodes['read'].data['label'].to(train_device)



    if train_junction_gtf_file is None or train_bam_files is None:
        logger.info("Without reference comments, the self-selected training set cannot be used. We will use the trained model.")
        models = torch.load("./trained_models.pth", map_location=train_device)
        model_f1_scores = torch.load("./model_f1_scores.pth", map_location=train_device)
    else:
        train_annotation_path = os.path.dirname(train_junction_gtf_file)
        train_junction_input_file = os.path.join(train_annotation_path,
                                           f"{os.path.splitext(os.path.basename(train_junction_gtf_file))[0]}.info")
        train_junction_df = process_junction(train_junction_gtf_file, train_junction_input_file)
        models = []
        model_f1_scores = []
        for alignment_bam_path in train_bam_files:
            train_bam_path = os.path.dirname(alignment_bam_path)
            # Add the suffix "_align" to the alignment.info file
            alignment_align_path = os.path.join(train_bam_path,
                                                f"{os.path.splitext(os.path.basename(alignment_bam_path))[0]}.info")

            # Call function
            feature_df, intron_df = train_preprocess(alignment_bam_path, alignment_align_path, train_junction_df)
            train_graph = train_consgraph(feature_df, intron_df)
            train_graph = train_graph[0].to(train_device)
            train_labels = train_graph.nodes['read'].data['label'].to(train_device)

            in_channels = train_graph.nodes['read'].data['feat'].shape[1]
            out_channels = 128
            epochs = 200
            model = BipartiteGCN(in_channels, out_channels).to(train_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

            # Train the model and save the best model
            logger.info(f"Training model...")
            best_model, best_val_f1 = train_model(model, train_graph, train_labels, val_graph, val_labels, optimizer, epochs=epochs, eps=0.1)

            models.append(best_model)
            model_f1_scores.append(best_val_f1)
    total_f1 = sum(model_f1_scores)
    weights = [f1 / total_f1 for f1 in model_f1_scores]

    return models, weights




# Model testing function
def test_model(models, weights, graph, test_device = 'cpu'):
    for model in models:
        model.to(test_device)
        model.eval()

    graph = graph.to(test_device)
    with torch.no_grad():
        all_probs = []
        for model in models:
            out = model(graph)
            probs = torch.nn.functional.softmax(out, dim=1)
            all_probs.append(probs)

        weighted_probs = sum(w * p for w, p in zip(weights, all_probs))
        predictions = torch.argmax(weighted_probs, dim=1)

        pred_labels = predictions.to(graph.device)
        read_id2s = graph.nodes['read'].data['read_id2']
        read_id2s = read_id2s.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()

        df = pd.DataFrame({
            'read_id2': read_id2s,
            'Predicted_Label': pred_labels
        })

        return df

def data_test(alignment_bam_path, models, weights, test_device):
    directory_path = os.path.dirname(alignment_bam_path)
    alignment_align_path = os.path.join(directory_path,
                                        f"{os.path.splitext(os.path.basename(alignment_bam_path))[0]}.info")

    feature_df, intron_df = test_preprocess(alignment_bam_path, alignment_align_path)
    graph = test_consgraph(feature_df, intron_df, device='cpu')
    predictions_df = test_model(models, weights, graph, test_device = test_device)
    return feature_df, predictions_df