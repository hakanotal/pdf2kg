import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import logging
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_knowledge_graph_visualization(graph_data_path, metadata_path=None, output_path="visualization.png", 
                                        figsize=(12, 10), progress=None):
    """
    Create a static visualization of the knowledge graph using matplotlib
    
    Args:
        graph_data_path: Path to the graph data CSV (finalgraph.csv)
        metadata_path: Path to the node metadata CSV (metadata.csv) - not used in this version
        output_path: Path to save the PNG visualization
        figsize: Size of the output figure in inches (width, height)
        progress: Optional progress callback for Gradio
        
    Returns:
        str: Path to the visualization PNG file
    """
    if progress:
        progress(0.1, desc="Loading graph data")
    
    try:
        # Load graph data
        graph_df = pd.read_csv(graph_data_path)
        logger.info(f"Loaded graph data with {len(graph_df)} edges")
        
        # Create a graph from the data
        if progress:
            progress(0.4, desc="Creating network graph")
        
        # Create an empty graph
        G = nx.Graph()
        
        # Track edge types for coloring
        edge_types = {}
        
        # Add nodes and edges to the graph
        for _, row in graph_df.iterrows():
            source = row['source']
            target = row['target']
            
            # Add nodes if they don't exist
            if source not in G.nodes:
                G.add_node(source, label=source)
            
            if target not in G.nodes:
                G.add_node(target, label=target)
            
            # Add the edge with weight
            weight = row.get('value', 1)
            edge_type = row.get('edge_type', 'default')
            edge_types[(source, target)] = edge_type
            
            G.add_edge(source, target, weight=weight, title=row.get('edge', ''))
        
        # Create the visualization
        if progress:
            progress(0.7, desc="Generating visualization")
        
        # Update output path to PNG if it's an HTML file
        if output_path.endswith('.html'):
            output_path = output_path.replace('.html', '.png')
            
        # Set up the figure
        plt.figure(figsize=figsize)
        
        # Use a spring layout for the graph
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # Calculate node sizes based on degree centrality (how many connections)
        degrees = dict(nx.degree(G))
        node_sizes = [30 + 50 * degrees[node] for node in G.nodes()]
        
        # Calculate edge weights
        edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
        edge_widths = [0.5 + 0.5 * w for w in edge_weights]
        
        # Set up edge colors
        edge_colors = []
        for u, v in G.edges():
            if edge_types.get((u, v)) == 'contextual_proximity':
                edge_colors.append('#808080')  # Gray for contextual proximity
            else:
                edge_colors.append('#22dd22')  # Green for direct relations
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        # Add title
        plt.title('Knowledge Graph Visualization', fontsize=16)
        plt.axis('off')  # Turn off the axis
        
        # Save the visualization
        if progress:
            progress(0.9, desc="Saving visualization")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
        
        if progress:
            progress(1.0, desc="Visualization complete")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        if progress:
            progress(0.0, desc=f"Error creating visualization: {e}")
        return None 