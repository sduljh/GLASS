import torch
import torch.nn as nn
from dgl.nn import GraphConv
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import dgl
import pandas as pd
import time

test_device = torch.device('cpu')
print(f'Testing on device: {test_device}')


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


def test_model(models, weights, graph, labels, output_file):
    for model in models:
        model.to(test_device)
        model.eval()

    graph = graph.to(test_device)
    labels = labels.to(test_device)

    with torch.no_grad():
        all_probs = []
        for model in models:
            out = model(graph)
            probs = torch.nn.functional.softmax(out, dim=1)
            all_probs.append(probs)

        weighted_probs = sum(w * p for w, p in zip(weights, all_probs))
        predictions = torch.argmax(weighted_probs, dim=1)

        accuracy = (predictions == labels).float().mean()
        f1 = f1_score(labels.cpu(), predictions.cpu(), average='weighted')
        conf_matrix = confusion_matrix(labels.cpu(), predictions.cpu())
        precision = precision_score(labels.cpu(), predictions.cpu(), average=None)
        recall = recall_score(labels.cpu(), predictions.cpu(), average=None)

        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        for i, (p, r) in enumerate(zip(precision, recall)):
            print(f'Class {i} - Precision: {p:.4f}, Recall: {r:.4f}')

        pred_labels = predictions.to(graph.device)
        read_id2s = graph.nodes['read'].data['read_id2']
        read_id2s = read_id2s.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()

        df = pd.DataFrame({
            'read_id2': read_id2s,
            'Predicted_Label': pred_labels
        })

        df.to_csv(output_file, sep='\t', index=False)
        print(f"Predictions have been saved to {output_file}")


def main():
    start_time = time.time()

    # 定义模型集合和对应的f1分数文件
    model_sets = [
        ('set1', './BipartiteGCN_models_set1.pth',
         './BipartiteGCN_model_f1_scores_set1.pth'),
        ('set2', './BipartiteGCN_models_set2.pth',
         './BipartiteGCN_model_f1_scores_set2.pth'),
        ('set3', './BipartiteGCN_models_set3.pth',
         './BipartiteGCN_model_f1_scores_set3.pth'),
        ('set4', './BipartiteGCN_models_set4.pth',
         './BipartiteGCN_model_f1_scores_set4.pth'),
        ('set5', './BipartiteGCN_models_set5.pth',
         './BipartiteGCN_model_f1_scores_set5.pth')
    ]

    test_fruits = ["SRR23513619"]

    for set_name, model_path, f1_scores_path in model_sets:
        print(f"\n{'=' * 50}")
        print(f"Processing model set: {set_name}")
        print(f"Loading models from: {model_path}")
        print(f"Loading F1 scores from: {f1_scores_path}")

        models = torch.load(model_path)
        model_f1_scores = torch.load(f1_scores_path)

        total_f1 = sum(model_f1_scores)
        weights = [f1 / total_f1 for f1 in model_f1_scores]

        for fruit in test_fruits:
            test_graph, _ = dgl.load_graphs(f'./{fruit}.dgl')
            test_graph = test_graph[0]
            test_labels = test_graph.nodes['read'].data['label']

            print(f"\nTesting {fruit} with {set_name} models...")
            output_file = f"./BipartiteGCN_test_predictions_{set_name}.csv"
            test_model(models, weights, test_graph, test_labels, output_file)

    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()