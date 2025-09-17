import ast
import torch
import dgl
import logging
from sklearn.preprocessing import StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Data loading function
def load_data(intron_info, feature_info):
    logger.info("Loading data...")    
    # Ensure 'matched_read_id2s' is in the correct format
    try:
        intron_info['matched_read_id2s'] = intron_info['matched_read_id2s'].apply(lambda x: list(x))
        logger.info("'matched_read_id2s' field successfully converted to list format.")
    except Exception as e:
        logger.error(f"Error while processing 'matched_read_id2s': {e}")
    
    reads_features = feature_info

    # Add index column for nodes
    reads_features['read_row_idx'] = reads_features.index
    # Add index column for hyperedges
    intron_info['intron_row_idx'] = intron_info.index

    logger.info(f"Data loaded successfully: {len(reads_features)} reads, {len(intron_info)} introns.")
    return intron_info, reads_features


# Build bipartite graph
def build_bipartite_graph(intron_info, feature_info):
    logger.info("Building bipartite graph...")
    
    num_reads = len(feature_info)
    num_introns = len(intron_info)

    # Build node and hyperedge indices in the graph
    graph = dgl.heterograph({
        ('read', 'to_intron', 'intron'): ([], []),
        ('intron', 'to_read', 'read'): ([], [])
    })

    graph.add_nodes(num_reads, ntype='read')  # Add original nodes
    graph.add_nodes(num_introns, ntype='intron')  # Add hyperedge nodes

    logger.info(f"Graph created with {graph.num_nodes('read')} 'read' nodes and {graph.num_nodes('intron')} 'intron' nodes.")
    return graph


# Extract and standardize node features
def extract_read_features(graph, feature_info, selected_columns, device):
    logger.info("Extracting and standardizing read features...")
    feature_matrix = feature_info[selected_columns].values
    # Standardize features
    try:
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(feature_matrix)
        feature_tensor = torch.tensor(standardized_features, dtype=torch.float32).to(device)
        logger.info("Features standardized successfully.")
    except Exception as e:
        logger.error(f"Error while standardizing features: {e}")
        raise e

    graph = graph.to(device)

    # Get labels and add them as node attributes
    try:
        labels_tensor = torch.tensor(feature_info['label'].values, dtype=torch.long).to(device)
        graph.nodes['read'].data['label'] = labels_tensor  # Add labels as node attributes
        logger.info("Labels added to graph nodes.")
    except Exception as e:
        logger.error(f"Error while adding labels: {e}")
        raise e

    try:
        read_id2_tensor = torch.tensor(feature_info['read_id2'].values, dtype=torch.long).to(device)
        graph.nodes['read'].data['feat'] = feature_tensor  # Add features to node
        graph.nodes['read'].data['read_id2'] = read_id2_tensor  # Add read_id2 to node
        logger.info(f"Node features and read_id2 added to graph: {feature_tensor.shape}")
    except Exception as e:
        logger.error(f"Error while adding node features: {e}")
        raise e

    return graph


# Add hyperedges to the graph
def add_intron_nodes(graph, intron_info, feature_info):
    logger.info("Adding intron nodes and edges to the graph...")

    # Convert feature_info['read_id2'] to set for faster lookup
    read_id2_set = set(feature_info['read_id2'].values)
    # Pre-build a mapping of read_id2 to read_row_idx (using a dictionary for faster lookup)
    read_id2_to_read_idx = dict(zip(feature_info['read_id2'], feature_info['read_row_idx']))

    # Initialize lists for storing edges
    src_list = []
    dst_list = []

    for intron_row_idx, row in enumerate(intron_info.itertuples(index=False)):
        logger.debug(f"Processing intron {intron_row_idx} with matched reads {row.matched_read_id2s}")
        for x in row.matched_read_id2s:  # No need to remove duplicates, just iterate
            if x in read_id2_set:  # Speed up lookup
                read_idx = read_id2_to_read_idx[x]  # Look up read_row_idx directly from dictionary
                src_list.append(read_idx)
                dst_list.append(intron_row_idx)
            else:
                logger.warning(f"Node {x} not found in feature_info")

    # The length of src_list and dst_list
    logger.info(f"Number of edges to add: {len(src_list)}")
    logger.info(f"Number of destination edges: {len(dst_list)}")

    # Add edges in batch to reduce number of calls
    try:
        graph.add_edges(src_list, dst_list, etype='to_intron')
        graph.add_edges(dst_list, src_list, etype='to_read')
        logger.info(f"Edges successfully added to the graph.")
    except Exception as e:
        logger.error(f"Error while adding edges: {e}")
        raise e

    return graph


def cons_graph(feature_df, intron_df, device='cpu'):
    logger.info("Starting training process...")

    # Load data
    intron_info, read_features = load_data(intron_df, feature_df)

    # Build bipartite graph
    graph = build_bipartite_graph(intron_info, read_features)

    # Extract and standardize node features, adding labels as node attributes
    selected_columns = [
        'read_id2_intron_number',
        'mapped-length_percent',
        'SOFT_CLIP_Left_percent',
        'SOFT_CLIP_right_percent',
        'CIGAR_score',
        'unique_intron_number_average',
        'unique_intron_number_1_average',
        'chaining_score',
        'start_proportion_average',
        'end_proportion_average',
    ]
    graph = extract_read_features(graph, read_features, selected_columns, device)
    # Add intron_nodes to the graph
    graph = add_intron_nodes(graph, intron_info, read_features)

    logger.info("Graph creation complete.")
    return graph



