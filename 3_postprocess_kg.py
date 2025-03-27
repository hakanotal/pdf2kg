import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import random
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from transformers import AutoModel, AutoTokenizer
import argparse
import json
import os


def load_knowledge_graph(filepath):
    """Load the knowledge graph from a CSV file."""
    return pd.read_csv(filepath)


def get_unique_nodes(graph_df):
    """Extract all unique nodes from the graph dataframe."""
    return pd.concat([graph_df['node_1'], graph_df['node_2']], axis=0).unique()


def load_language_model():
    """Load the pre-trained language model for node embedding."""
    model_name = "Alibaba-NLP/gte-base-en-v1.5"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def compute_node_embeddings(nodes, model, tokenizer):
    """Compute embeddings for all nodes using the language model."""
    embeddings = []
    print("Computing node embeddings...")
    for node in tqdm(nodes):
        inputs = tokenizer(node, return_tensors="pt", max_length=128, truncation=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # extract the embedding of the [CLS] token
        embeddings.append(embedding.detach().numpy())
    return np.array(embeddings)


def cluster_nodes(node_embeddings, nodes, eps=0.1, min_samples=1):
    """Cluster similar nodes using DBSCAN."""
    print("Clustering similar nodes...")
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(node_embeddings.squeeze(axis=1))
    
    node_to_label = {node: label for node, label in zip(nodes, cluster_labels)}
    
    cluster_to_nodes = {}
    for node, label in zip(nodes, cluster_labels):
        if label not in cluster_to_nodes:
            cluster_to_nodes[label] = []
        cluster_to_nodes[label].append(node)
    
    # Print clusters with multiple nodes
    for label, nodes_in_cluster in cluster_to_nodes.items():
        if len(nodes_in_cluster) > 1:
            print(f"Cluster {label}: {nodes_in_cluster}")
    
    # For each cluster, select the most common node as representative
    label_to_node = {}
    for label, nodes_in_cluster in cluster_to_nodes.items():
        node_counts = {}
        for node in nodes_in_cluster:
            node_counts[node] = list(nodes).count(node)
        most_common_node = max(node_counts, key=node_counts.get)
        label_to_node[label] = most_common_node
    
    return node_to_label, label_to_node, cluster_to_nodes


def create_node_types_df(graph_df):
    """Create a dataframe with node types information."""
    df_node_types = graph_df[['node_1','node_1_type']].rename(columns={'node_1': 'node', 'node_1_type': 'node_type'})
    df_node_types = df_node_types._append(graph_df[['node_2','node_2_type']].rename(columns={'node_2': 'node', 'node_2_type': 'node_type'}))
    df_node_types.drop_duplicates(subset=['node'], inplace=True)
    return df_node_types


def create_merged_graph(graph_df, nodes, node_to_label, label_to_node, df_node_types):
    """Create a new graph with merged nodes."""
    print("Creating merged graph...")
    G_merged = nx.Graph()
    
    # Add nodes to the graph
    for node in nodes:
        label = node_to_label[node]
        new_node = label_to_node[label]

        if new_node not in G_merged.nodes:
            G_merged.add_node(
                str(new_node),
                node_label=str(new_node),
                node_type=df_node_types[df_node_types['node'] == node]['node_type'].iloc[0]
            )

    # Add edges to the graph
    for index, row in graph_df.iterrows():
        n1 = label_to_node[node_to_label[str(row["node_1"])]]
        n2 = label_to_node[node_to_label[str(row["node_2"])]]
        if n1 != n2:
            G_merged.add_edge(
                str(n1),
                str(n2),
                edge_title=row["edge_type"],
                edge_details=row["edge"],
                weight=row['count']/4,
                ref=row["chunk_id"]
            )
    
    return G_merged


def colors2NodeTypes(nodes, df_node_types, palette="hls"):
    """Assign colors to nodes based on node types."""
    node_types = df_node_types['node_type'].unique()
    p = sns.color_palette(palette, len(node_types)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0

    for node_type in node_types:
        color = p.pop()
        group += 1
        for node in nodes:
            if df_node_types[df_node_types['node'] == node]['node_type'].iloc[0] == node_type:
                rows += [{"node": node, "color": color, "group": group}]

    df_colors = pd.DataFrame(rows)
    return df_colors


def update_graph_with_merged_nodes(graph_df, node_to_label, label_to_node):
    """Update the graph dataframe with merged nodes."""
    print("Updating graph with merged nodes...")
    # Create a temporary dataframe with all nodes
    dfg_temp = graph_df[['node_1']].rename(columns={'node_1': 'node'})
    dfg_temp = dfg_temp._append(graph_df[['node_2']].rename(columns={'node_2': 'node'}))
    dfg_temp.drop_duplicates(subset=['node'], inplace=True)

    # Update node references in the graph
    for node in list(dfg_temp['node']):
        label = node_to_label[node]
        new_node = label_to_node[label]

        for index, row in graph_df[graph_df['node_1'] == node].iterrows():
            graph_df.at[index, 'node_1'] = new_node

        for index, row in graph_df[graph_df['node_2'] == node].iterrows():
            graph_df.at[index, 'node_2'] = new_node
    
    return graph_df


def create_final_graph(graph_df, output_file="finalgraph.csv"):
    """Create and save the final graph to CSV."""
    print(f"Creating final graph and saving to {output_file}...")
    final_df = graph_df.copy()[['node_1', 'node_2', 'edge', 'edge_type', 'count']]
    final_df.rename(columns={'node_1': 'source', 'node_2': 'target', 'count': 'value'}, inplace=True)
    final_df['color'] = final_df['edge_type'].apply(lambda x: '#808080' if x == 'contextual_proximity' else '#22dd22')
    final_df.to_csv(output_file, sep=",", index=False)
    return final_df


def create_node_metadata(df_node_types, output_file="metadata.csv"):
    """Create and save metadata about nodes to CSV."""
    print(f"Creating node metadata and saving to {output_file}...")
    
    # Create a new dataframe with the required columns
    metadata_df = df_node_types.copy()
    metadata_df.rename(columns={'node': 'id', 'node_type': 'type'}, inplace=True)
    
    # Create a type_id column (numeric identifier for each type)
    type_mapping = {t: i for i, t in enumerate(metadata_df['type'].unique())}
    metadata_df['type_id'] = metadata_df['type'].map(type_mapping)
    
    # Save to CSV
    metadata_df.to_csv(output_file, sep=",", index=False)
    return metadata_df


def postprocess_knowledge_graph(kg_path, output_path):
    """
    Post-process the knowledge graph to improve quality
    
    Args:
        kg_path: Path to the input knowledge graph file
        output_path: Path to save the processed knowledge graph
    """
    # Load the knowledge graph
    with open(kg_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    
    # Implement post-processing steps
    # 1. Clean data
    # 2. Resolve entity references
    # 3. Remove duplicates
    # 4. Other optimization steps
    
    # This is a placeholder for your actual post-processing logic
    processed_kg = kg  # Replace with actual processing
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed knowledge graph
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_kg, f, indent=2)
    
    print(f"Processed knowledge graph saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Post-process knowledge graph.')
    parser.add_argument('--kg', required=True, help='Path to the input knowledge graph file')
    parser.add_argument('--output', required=True, help='Path to save the processed knowledge graph')
    
    args = parser.parse_args()
    
    postprocess_knowledge_graph(args.kg, args.output)


if __name__ == "__main__":
    main()