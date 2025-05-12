import os
import json
import torch
import random
import logging
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import DBSCAN
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    logger.warning("GLiNER is not available. Entity classification will be skipped.")

def load_knowledge_graph(filepath):
    """Load the knowledge graph from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert the edges to a DataFrame
    edges = data.get('edges', [])
    if not edges:
        logger.error(f"No edges found in knowledge graph file: {filepath}")
        return None
    
    return pd.DataFrame(edges)

def get_unique_nodes(graph_df):
    """Extract all unique nodes from the graph dataframe."""
    return pd.concat([graph_df['node_1'], graph_df['node_2']], axis=0).unique()

def load_language_model(progress=None):
    """Load the pre-trained language model for node embedding."""
    if progress:
        progress(0.1, desc="Loading language model")
    
    model_name = "Alibaba-NLP/gte-base-en-v1.5"
    
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Error loading language model: {e}")
        if progress:
            progress(0.0, desc=f"Error loading language model: {e}")
        return None, None
    
    return model, tokenizer

def compute_node_embeddings(nodes, model, tokenizer, progress=None):
    """Compute embeddings for all nodes using the language model."""
    embeddings = []
    logger.info("Computing node embeddings...")
    
    if progress:
        progress(0.2, desc="Computing node embeddings")
    
    # Use tqdm for progress tracking in console
    for i, node in enumerate(tqdm(nodes)):
        # Update progress in UI if available
        if progress and i % 10 == 0:  # Update every 10 nodes to avoid UI slowdown
            current_progress = 0.2 + (0.3 * (i / len(nodes)))
            progress(current_progress, desc=f"Computing embeddings: {i+1}/{len(nodes)}")
        
        try:
            inputs = tokenizer(node, return_tensors="pt", max_length=128, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()  # extract the embedding of the [CLS] token
            embeddings.append(embedding)
        except Exception as e:
            logger.error(f"Error computing embedding for node '{node}': {e}")
            # Use zeros as a fallback
            embeddings.append(np.zeros((1, model.config.hidden_size)))
    
    return np.array(embeddings)

def cluster_nodes(node_embeddings, nodes, eps=0.1, min_samples=1, progress=None):
    """Cluster similar nodes using DBSCAN."""
    logger.info("Clustering similar nodes...")
    
    if progress:
        progress(0.5, desc="Clustering similar nodes")
    
    try:
        # Reshape embeddings for clustering
        if len(node_embeddings.shape) == 3:
            # If we have a 3D array [n_nodes, 1, embedding_dim]
            embeddings_2d = node_embeddings.squeeze(axis=1)
        else:
            # Already in the right shape
            embeddings_2d = node_embeddings
            
        # Run DBSCAN clustering
        cluster_labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embeddings_2d)
        
        node_to_label = {node: label for node, label in zip(nodes, cluster_labels)}
        
        cluster_to_nodes = {}
        for node, label in zip(nodes, cluster_labels):
            if label not in cluster_to_nodes:
                cluster_to_nodes[label] = []
            cluster_to_nodes[label].append(node)
        
        # Print clusters with multiple nodes
        for label, nodes_in_cluster in cluster_to_nodes.items():
            if len(nodes_in_cluster) > 1:
                logger.info(f"Cluster {label}: {nodes_in_cluster}")
        
        # For each cluster, select the most common node as representative
        label_to_node = {}
        for label, nodes_in_cluster in cluster_to_nodes.items():
            node_counts = {}
            for node in nodes_in_cluster:
                node_counts[node] = list(nodes).count(node)
            most_common_node = max(node_counts, key=node_counts.get)
            label_to_node[label] = most_common_node
        
        return node_to_label, label_to_node, cluster_to_nodes
    except Exception as e:
        logger.error(f"Error clustering nodes: {e}")
        if progress:
            progress(0.5, desc=f"Error clustering nodes: {e}")
        return None, None, None

def create_node_types_df(graph_df):
    """Create a dataframe with node types information."""
    # Get all unique nodes from the graph
    nodes_set = set()
    for node in graph_df['node_1']:
        nodes_set.add(node)
    for node in graph_df['node_2']:
        nodes_set.add(node)
    
    # Create dataframe with node information
    return pd.DataFrame(list(nodes_set), columns=['node'])

def create_merged_graph(graph_df, nodes, node_to_label, label_to_node, df_node_types, progress=None):
    """Create a new graph with merged nodes."""
    logger.info("Creating merged graph...")
    
    if progress:
        progress(0.7, desc="Creating merged graph")
    
    G_merged = nx.Graph()
    
    # Add nodes to the graph
    for node in nodes:
        label = node_to_label[node]
        new_node = label_to_node[label]

        if new_node not in G_merged.nodes:
            G_merged.add_node(
                str(new_node),
                node_label=str(new_node)
            )

    # Add edges to the graph
    for index, row in graph_df.iterrows():
        try:
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
        except Exception as e:
            logger.warning(f"Error adding edge {row['node_1']} -> {row['node_2']}: {e}")
    
    return G_merged

def update_graph_with_merged_nodes(graph_df, node_to_label, label_to_node, progress=None):
    """Update the graph dataframe with merged nodes."""
    logger.info("Updating graph with merged nodes...")
    
    if progress:
        progress(0.8, desc="Updating graph with merged nodes")
    
    # Create a temporary dataframe with all nodes
    dfg_temp = graph_df[['node_1']].rename(columns={'node_1': 'node'})
    dfg_temp = pd.concat([dfg_temp, graph_df[['node_2']].rename(columns={'node_2': 'node'})])
    dfg_temp.drop_duplicates(subset=['node'], inplace=True)

    # Update node references in the graph
    for node in list(dfg_temp['node']):
        try:
            label = node_to_label[node]
            new_node = label_to_node[label]

            for index, row in graph_df[graph_df['node_1'] == node].iterrows():
                graph_df.at[index, 'node_1'] = new_node

            for index, row in graph_df[graph_df['node_2'] == node].iterrows():
                graph_df.at[index, 'node_2'] = new_node
        except Exception as e:
            logger.warning(f"Error updating node {node}: {e}")
    
    return graph_df

def create_final_graph(graph_df, output_file="finalgraph.csv"):
    """Create and save the final graph to CSV."""
    logger.info(f"Creating final graph and saving to {output_file}...")
    
    final_df = graph_df.copy()[['node_1', 'node_2', 'edge', 'edge_type', 'count']]
    final_df.rename(columns={'node_1': 'source', 'node_2': 'target', 'count': 'value'}, inplace=True)
    final_df['color'] = final_df['edge_type'].apply(lambda x: '#808080' if x == 'contextual_proximity' else '#22dd22')
    final_df.to_csv(output_file, sep=",", index=False)
    return final_df

def create_node_metadata(df_node_types, output_file="metadata.csv"):
    """Create and save metadata about nodes to CSV."""
    logger.info(f"Creating node metadata and saving to {output_file}...")
    
    # Create a new dataframe with just node IDs
    metadata_df = df_node_types.copy()
    metadata_df.rename(columns={'node': 'id'}, inplace=True)
    
    # Save to CSV
    metadata_df.to_csv(output_file, sep=",", index=False)
    return metadata_df

def classify_entities(nodes, config, progress=None):
    """
    Classify nodes using GLiNER named entity recognition model
    
    Args:
        nodes: List of nodes to classify
        config: Configuration dictionary with entity_classification settings
        progress: Optional progress callback
        
    Returns:
        Dictionary mapping node text to entity label
    """
    if not GLINER_AVAILABLE:
        logger.warning("GLiNER is not available. Entity classification will be skipped.")
        return {}
    
    if progress:
        progress(0.85, desc="Classifying entities")
    
    try:
        model_name = config.get("entity_classification", {}).get("model", "gliner-community/gliner_large-v2.5")
        entity_labels = config.get("entity_classification", {}).get("entity_labels", 
                                  ["technique", "tool", "person", "organization", "concept", "resource"])
        
        logger.info(f"Loading GLiNER model: {model_name}")
        model = GLiNER.from_pretrained(model_name, load_tokenizer=True)
        
        # Process nodes in batches to avoid memory issues
        node_to_entity = {}
        batch_size = 10
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i+batch_size]
            
            # Update progress
            if progress:
                current_progress = 0.85 + (0.14 * (i / len(nodes)))
                progress(current_progress, desc=f"Classifying entities: {i}/{len(nodes)}")
            
            # Process each node individually for more accurate classification
            for node in batch:
                try:
                    # Only classify non-empty strings with length > 1
                    if isinstance(node, str) and len(node.strip()) > 1:
                        entities = model.predict_entities(node, entity_labels)
                        # If the entire text is classified as an entity, use that label
                        if entities and entities[0]["text"].lower() == node.lower():
                            node_to_entity[node] = entities[0]["label"]
                        # Otherwise, use the first entity found or default to "concept"
                        elif entities:
                            node_to_entity[node] = entities[0]["label"]
                        else:
                            node_to_entity[node] = "concept"  # Default category
                    else:
                        node_to_entity[node] = "concept"  # Default for empty or very short nodes
                except Exception as e:
                    logger.warning(f"Error classifying node '{node}': {e}")
                    node_to_entity[node] = "concept"  # Default category on error
        
        logger.info(f"Classified {len(node_to_entity)} entities")
        return node_to_entity
    
    except Exception as e:
        logger.error(f"Error in entity classification: {e}")
        return {}

def update_metadata_with_entity_types(metadata_df, node_to_entity, output_file="metadata.csv"):
    """Add entity type information to the metadata DataFrame and save it."""
    logger.info(f"Adding entity types to metadata and saving to {output_file}...")
    
    # Add entity type column
    metadata_df['entity_type'] = metadata_df['id'].map(lambda x: node_to_entity.get(x, "concept"))
    
    # Save to CSV
    metadata_df.to_csv(output_file, sep=",", index=False)
    return metadata_df

def postprocess_knowledge_graph(kg_path, output_path, progress=None, config=None):
    """
    Post-process the knowledge graph to improve quality
    
    Args:
        kg_path: Path to the input knowledge graph file
        output_path: Path to save the processed knowledge graph
        progress: Optional progress callback for Gradio
        config: Optional configuration dictionary
        
    Returns:
        tuple: (output_path, combined_nodes_info) where:
            - output_path: Path to the processed output directory
            - combined_nodes_info: Information about combined nodes during clustering
    """
    # Load configuration if not provided
    if config is None:
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using default values.")
            config = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Set up file paths
    final_graph_path = os.path.join(output_path, "finalgraph.csv")
    metadata_path = os.path.join(output_path, "metadata.csv")
    visualization_path = os.path.join(output_path, "visualization.html")
    
    # Store information about combined nodes
    combined_nodes_info = {}
    
    # Load the knowledge graph
    try:
        graph_df = load_knowledge_graph(kg_path)
        if graph_df is None or len(graph_df) == 0:
            logger.error(f"Failed to load knowledge graph from {kg_path}")
            return None, None
        
        logger.info(f"Loaded knowledge graph with {len(graph_df)} edges")
    except Exception as e:
        logger.error(f"Error loading knowledge graph: {e}")
        return None, None
    
    # Get unique nodes
    nodes = get_unique_nodes(graph_df)
    logger.info(f"Found {len(nodes)} unique nodes")
    
    # Load language model for embeddings
    model, tokenizer = load_language_model(progress)
    if model is None or tokenizer is None:
        logger.error("Failed to load language model, skipping embedding-based clustering")
        # Create a simple output without clustering
        df_node_types = create_node_types_df(graph_df)
        create_final_graph(graph_df, final_graph_path)
        
        # Classify entities for metadata
        node_to_entity = classify_entities(list(df_node_types['node']), config, progress)
        metadata_df = create_node_metadata(df_node_types, metadata_path)
        update_metadata_with_entity_types(metadata_df, node_to_entity, metadata_path)
        
        return output_path, None
    
    # Compute node embeddings
    node_embeddings = compute_node_embeddings(nodes, model, tokenizer, progress)
    
    # Cluster similar nodes
    node_to_label, label_to_node, cluster_to_nodes = cluster_nodes(
        node_embeddings, nodes, eps=0.1, min_samples=1, progress=progress
    )
    
    if node_to_label is None:
        logger.error("Failed to cluster nodes, skipping clustering")
        # Create a simple output without clustering
        df_node_types = create_node_types_df(graph_df)
        create_final_graph(graph_df, final_graph_path)
        
        # Classify entities for metadata
        node_to_entity = classify_entities(list(df_node_types['node']), config, progress)
        metadata_df = create_node_metadata(df_node_types, metadata_path)
        update_metadata_with_entity_types(metadata_df, node_to_entity, metadata_path)
        
        return output_path, None
    
    # Collect information about combined nodes
    for label, nodes_in_cluster in cluster_to_nodes.items():
        if len(nodes_in_cluster) > 1:
            representative_node = label_to_node[label]
            combined_nodes_info[representative_node] = [n for n in nodes_in_cluster if n != representative_node]
    
    # Create node types dataframe
    df_node_types = create_node_types_df(graph_df)
    
    # Create merged graph
    G_merged = create_merged_graph(graph_df, nodes, node_to_label, label_to_node, df_node_types, progress)
    
    # Update graph with merged nodes
    updated_graph_df = update_graph_with_merged_nodes(graph_df, node_to_label, label_to_node, progress)
    
    # Create and save final graph
    create_final_graph(updated_graph_df, final_graph_path)
    
    # Create metadata and classify entities
    metadata_df = create_node_metadata(df_node_types, metadata_path)
    
    # Classify entities
    node_to_entity = classify_entities(list(metadata_df['id']), config, progress)
    update_metadata_with_entity_types(metadata_df, node_to_entity, metadata_path)
    
    if progress:
        progress(1.0, desc="Post-processing complete")
    
    logger.info(f"Post-processed knowledge graph saved to {output_path}")
    
    # Format combined nodes info for display
    combined_nodes_text = ""
    if combined_nodes_info:
        combined_nodes_text = "Redundant nodes eliminated:\n\n"
        for main_node, merged_nodes in combined_nodes_info.items():
            if merged_nodes:
                quoted_nodes = ["'" + n + "'" for n in merged_nodes]
                combined_nodes_text += f"• '{main_node}' now represents: {', '.join(quoted_nodes)}\n"
    
    # Add entity classification summary
    if node_to_entity:
        entity_summary = {}
        for node, entity_type in node_to_entity.items():
            if entity_type not in entity_summary:
                entity_summary[entity_type] = 0
            entity_summary[entity_type] += 1
        
        combined_nodes_text += "\n\nEntity classification summary:\n\n"
        for entity_type, count in entity_summary.items():
            combined_nodes_text += f"• {entity_type}: {count} nodes\n"
    
    return output_path, combined_nodes_text 