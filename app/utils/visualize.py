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
import matplotlib.patches as mpatches

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_knowledge_graph_visualization(graph_data_path, metadata_path=None, output_path="visualization.png", 
                                        figsize=(12, 10), progress=None, show_contextual_proximity=True):
    """
    Create a static visualization of the knowledge graph using matplotlib
    
    Args:
        graph_data_path: Path to the graph data CSV (finalgraph.csv)
        metadata_path: Path to the node metadata CSV (metadata.csv) with entity type information
        output_path: Path to save the PNG visualization
        figsize: Size of the output figure in inches (width, height)
        progress: Optional progress callback for Gradio
        show_contextual_proximity: Toggle to show/hide contextual_proximity edges
        
    Returns:
        str: Path to the visualization PNG file
    """
    if progress:
        progress(0.1, desc="Loading graph data")
    
    try:
        # Load graph data
        graph_df = pd.read_csv(graph_data_path)
        logger.info(f"Loaded graph data with {len(graph_df)} edges")
        
        # Load metadata if available
        node_entity_types = {}
        entity_type_colors = {
            'technique': '#1f77b4',  # Blue
            'tool': '#ff7f0e',       # Orange
            'person': '#2ca02c',     # Green
            'organization': '#d62728', # Red
            'concept': '#9467bd',    # Purple
            'resource': '#8c564b',   # Brown
            'default': '#17becf'     # Cyan
        }
        
        if metadata_path and os.path.exists(metadata_path):
            try:
                metadata_df = pd.read_csv(metadata_path)
                if 'id' in metadata_df.columns and 'entity_type' in metadata_df.columns:
                    node_entity_types = dict(zip(metadata_df['id'], metadata_df['entity_type']))
                    logger.info(f"Loaded metadata with entity types for {len(node_entity_types)} nodes")
                else:
                    logger.warning("Metadata file does not contain required columns (id, entity_type)")
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
        
        # Create a graph from the data
        if progress:
            progress(0.4, desc="Creating network graph")
        
        # Create an empty graph
        G = nx.Graph()
        
        # Track edge types and edge labels for coloring and labeling
        edge_types = {}
        edge_labels = {}
        
        # Add nodes and edges to the graph
        for _, row in graph_df.iterrows():
            source = row['source']
            target = row['target']
            edge_type = row.get('edge_type', 'default')
            
            # Skip contextual_proximity edges if toggle is off
            if not show_contextual_proximity and edge_type == 'contextual_proximity':
                continue
            
            # Add nodes if they don't exist, with entity type from metadata
            if source not in G.nodes:
                entity_type = node_entity_types.get(source, 'default')
                G.add_node(source, label=source, entity_type=entity_type)
            
            if target not in G.nodes:
                entity_type = node_entity_types.get(target, 'default')
                G.add_node(target, label=target, entity_type=entity_type)
            
            # Add the edge with weight
            weight = row.get('value', 1)
            edge_types[(source, target)] = edge_type
            
            # Store edge label if it's a relation edge
            if edge_type == 'relation':
                edge_labels[(source, target)] = row.get('edge', '')
            
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
        
        # Set up edge colors and edge widths based on type
        edge_colors = []
        edge_widths = []
        
        for u, v in G.edges():
            weight = G[u][v].get('weight', 1)
            if edge_types.get((u, v)) == 'contextual_proximity':
                edge_colors.append('#808080')  # Gray for contextual proximity
                edge_widths.append(0.5 + 0.3 * weight)  # Thinner for contextual proximity
            else:
                edge_colors.append('#22dd22')  # Green for direct relations
                edge_widths.append(1.0 + 0.5 * weight)  # Thicker for relations
        
        # Node colors based on entity type
        node_colors = []
        
        for node in G.nodes():
            entity_type = G.nodes[node].get('entity_type', 'default')
            color = entity_type_colors.get(entity_type, entity_type_colors['default'])
            node_colors.append(color)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        # Draw edge labels for relation edges
        if not show_contextual_proximity or len(edge_labels) > 0:
            # Calculate better label positions by slightly offsetting labels from the edges
            edge_label_pos = {}
            for (u, v), label in edge_labels.items():
                # Truncate long labels for clarity
                if len(label) > 15:
                    edge_labels[(u,v)] = label[:15] + "..."
                
                # Calculate position for labels to avoid overlap when possible
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                # Add a small offset perpendicular to the edge
                edge_label_pos[(u, v)] = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            nx.draw_networkx_edge_labels(
                G, pos, 
                edge_labels=edge_labels,
                font_size=7,
                font_color='black',
                alpha=0.8,
                rotate=False,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            )
        
        # Add legend for edges and entity types
        legend_items = []
        
        # Only add contextual proximity to legend if we're showing them
        if show_contextual_proximity:
            contextual_patch = mpatches.Patch(color='#808080', label='Contextual Proximity', alpha=0.5)
            legend_items.append(contextual_patch)
        
        relation_patch = mpatches.Patch(color='#22dd22', label='Direct Relation', alpha=0.5)
        legend_items.append(relation_patch)
        
        # Add entity type legend items if metadata is available
        if node_entity_types:
            # Get unique entity types
            unique_entity_types = set(node_entity_types.values())
            
            # Add legend items for each entity type
            for entity_type in sorted(unique_entity_types):
                color = entity_type_colors.get(entity_type, entity_type_colors['default'])
                entity_patch = mpatches.Patch(color=color, label=f'{entity_type}', alpha=0.7)
                legend_items.append(entity_patch)
        
        plt.legend(handles=legend_items, loc='upper right', fontsize=10)
        
        # Add title
        title_text = 'Knowledge Graph Visualization'
        if not show_contextual_proximity:
            title_text += ' (Relations Only)'
        plt.title(title_text, fontsize=16)
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