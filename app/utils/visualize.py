import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import logging
import matplotlib.patches as mpatches
import yaml
from matplotlib.cm import get_cmap
from pyvis.network import Network

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load the application configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Return default entity labels if config can't be loaded
        return {"entity_classification": {"entity_labels": []}}

def generate_entity_colors(entity_types):
    """Generate colors for entity types using matplotlib color maps
    
    Args:
        entity_types: List of entity type names
        
    Returns:
        Dictionary mapping entity types to color codes
    """
    # Choose a colormap that gives distinct colors
    colormap = get_cmap('tab20')  # 20 distinct colors
    
    # If we have more entity types than colors in tab20, fall back to other maps
    if len(entity_types) > 20:
        colormap = get_cmap('viridis')
    
    # Generate colors
    entity_type_colors = {}
    
    # Always add default first
    entity_type_colors['default'] = '#17becf'  # Keep a recognizable color for default
    
    # Generate colors for each entity type
    for i, entity_type in enumerate(entity_types):
        # Normalize index to be between 0 and 1 for color mapping
        normalized_idx = i / max(1, len(entity_types) - 1)
        
        if len(entity_types) <= 20:
            # For tab20, use discrete colors
            color_idx = i % 20
            rgba_color = colormap(color_idx)
        else:
            # For continuous colormaps like viridis
            rgba_color = colormap(normalized_idx)
        
        # Convert RGBA to hex
        hex_color = matplotlib.colors.rgb2hex(rgba_color)
        entity_type_colors[entity_type] = hex_color
    
    return entity_type_colors

def filter_relations_only(graph_data_path, output_dir=None):
    """
    Filter graph data to include only 'relation' type edges and save to a new CSV file.
    
    Args:
        graph_data_path: Path to the original graph data CSV 
        output_dir: Directory to save the filtered CSV (defaults to same directory as input)
        
    Returns:
        str: Path to the filtered CSV file
    """
    try:
        # Load the graph data
        graph_df = pd.read_csv(graph_data_path)
        logger.info(f"Loaded graph data with {len(graph_df)} edges for filtering")
        
        # Filter for relation edges only
        if 'edge_type' in graph_df.columns:
            relations_df = graph_df[graph_df['edge_type'] == 'relation'].copy()
            logger.info(f"Filtered to {len(relations_df)} relation edges")
        else:
            # If edge_type column doesn't exist, assume all are relations
            relations_df = graph_df.copy()
            logger.warning("No edge_type column found, assuming all edges are relations")
        
        # Determine output path
        if output_dir is None:
            output_dir = os.path.dirname(graph_data_path)
            
        # Generate the output filename
        base_filename = os.path.basename(graph_data_path)
        filename_without_ext = os.path.splitext(base_filename)[0]
        relations_filename = f"{filename_without_ext}_relations_only.csv"
        output_path = os.path.join(output_dir, relations_filename)
        
        # Save the filtered data
        relations_df.to_csv(output_path, index=False)
        logger.info(f"Saved relations-only graph to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error filtering relations: {e}")
        return None

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
        
        # Also create a relations-only CSV file (but don't use it for visualization)
        filter_relations_only(graph_data_path)
        
        # Load entity types from config
        config = load_config()
        entity_labels = config.get('entity_classification', {}).get('entity_labels', [])
        logger.info(f"Loaded {len(entity_labels)} entity types from config")
        
        # Generate colors for entity types
        entity_type_colors = generate_entity_colors(entity_labels)
        
        # Load metadata if available
        node_entity_types = {}
        
        if metadata_path and os.path.exists(metadata_path):
            try:
                metadata_df = pd.read_csv(metadata_path)
                if 'id' in metadata_df.columns and 'entity_type' in metadata_df.columns:
                    node_entity_types = dict(zip(metadata_df['id'], metadata_df['entity_type']))
                    logger.info(f"Loaded metadata with entity types for {len(node_entity_types)} nodes")
                    
                    # Add any entity types from metadata that weren't in config
                    unique_entity_types = set(node_entity_types.values())
                    for entity_type in unique_entity_types:
                        if entity_type not in entity_type_colors and entity_type != 'default':
                            # Generate new colors for entity types found in data but not in config
                            new_types = [et for et in unique_entity_types if et not in entity_type_colors and et != 'default']
                            new_colors = generate_entity_colors(new_types)
                            entity_type_colors.update(new_colors)
                            logger.info(f"Added colors for {len(new_colors)} additional entity types found in metadata")
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
        
        # Add separated legends for edge types and entity types
        
        # Edge type legend items
        edge_legend_items = []
        if show_contextual_proximity:
            contextual_patch = mpatches.Patch(color='#808080', label='Contextual Proximity', alpha=0.5)
            edge_legend_items.append(contextual_patch)
        
        relation_patch = mpatches.Patch(color='#22dd22', label='Direct Relation', alpha=0.5)
        edge_legend_items.append(relation_patch)
        
        # Entity type legend items
        entity_legend_items = []
        if node_entity_types:
            # Get unique entity types that are actually used in the graph
            unique_entity_types = set(nx.get_node_attributes(G, 'entity_type').values())
            
            # Add legend items for each entity type
            for entity_type in sorted(unique_entity_types):
                if entity_type in entity_type_colors:
                    color = entity_type_colors[entity_type]
                    entity_patch = mpatches.Patch(color=color, label=f'{entity_type}', alpha=0.7)
                    entity_legend_items.append(entity_patch)
        
        # Create two separate legends with titles
        if edge_legend_items:
            first_legend = plt.legend(handles=edge_legend_items, loc='upper left', 
                          title='Edge Types', fontsize=9, title_fontsize=10)
            plt.gca().add_artist(first_legend)  # Add first legend and keep it
        
        if entity_legend_items:
            # Place entity types legend in a different location
            plt.legend(handles=entity_legend_items, loc='upper right', 
                      title='Entity Types', fontsize=9, title_fontsize=10)
        
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

def create_interactive_knowledge_graph(graph_data_path, metadata_path=None, output_path="visualization.html", 
                                      progress=None, show_contextual_proximity=True):
    """
    Create an interactive visualization of the knowledge graph using pyvis
    
    Args:
        graph_data_path: Path to the graph data CSV (finalgraph.csv)
        metadata_path: Path to the node metadata CSV (metadata.csv) with entity type information
        output_path: Path to save the HTML visualization
        progress: Optional progress callback for Gradio
        show_contextual_proximity: Toggle to show/hide contextual_proximity edges
        
    Returns:
        str: HTML string containing the interactive visualization
    """
    if progress:
        progress(0.1, desc="Loading graph data")
    
    try:
        # Load graph data
        graph_df = pd.read_csv(graph_data_path)
        logger.info(f"Loaded graph data with {len(graph_df)} edges for interactive visualization")
        
        # Load entity types from config
        config = load_config()
        entity_labels = config.get('entity_classification', {}).get('entity_labels', [])
        logger.info(f"Loaded {len(entity_labels)} entity types from config")
        
        # Generate colors for entity types
        entity_type_colors = generate_entity_colors(entity_labels)
        
        # Load metadata if available
        node_entity_types = {}
        
        if metadata_path and os.path.exists(metadata_path):
            try:
                metadata_df = pd.read_csv(metadata_path)
                if 'id' in metadata_df.columns and 'entity_type' in metadata_df.columns:
                    node_entity_types = dict(zip(metadata_df['id'], metadata_df['entity_type']))
                    logger.info(f"Loaded metadata with entity types for {len(node_entity_types)} nodes")
                    
                    # Add any entity types from metadata that weren't in config
                    unique_entity_types = set(node_entity_types.values())
                    for entity_type in unique_entity_types:
                        if entity_type not in entity_type_colors and entity_type != 'default':
                            new_types = [et for et in unique_entity_types if et not in entity_type_colors and et != 'default']
                            new_colors = generate_entity_colors(new_types)
                            entity_type_colors.update(new_colors)
                else:
                    logger.warning("Metadata file does not contain required columns (id, entity_type)")
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
        
        # Create an empty NetworkX graph
        if progress:
            progress(0.4, desc="Creating network graph")
        
        G = nx.Graph()
        
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
                G.add_node(source, label=source, title=source, group=entity_type)
            
            if target not in G.nodes:
                entity_type = node_entity_types.get(target, 'default')
                G.add_node(target, label=target, title=target, group=entity_type)
            
            # Add the edge with weight and title
            weight = row.get('value', 1)
            edge_label = row.get('edge', '')
            G.add_edge(source, target, weight=weight, title=edge_label, edge_type=edge_type)
        
        # Create the pyvis Network
        if progress:
            progress(0.7, desc="Generating interactive visualization")
            
        # Create pyvis network with better options for visibility
        nt = Network(height="700px", width="100%", notebook=False, directed=False)
        
        # Set physics options for better layout
        nt.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.001, damping=0.09)
        
        # Configure global node options for better visibility
        nt.set_options("""
        {
          "nodes": {
            "font": {
              "size": 8,
              "face": "arial",
              "bold": true
            },
            "scaling": {
              "label": {
                "enabled": true,
                "min": 14,
                "max": 24
              }
            }
          },
          "edges": {
            "font": {
              "size": 12,
              "align": "middle"
            },
            "smooth": {
              "type": "continuous",
              "forceDirection": "none"
            },
            "arrows": {
              "to": {
                "enabled": true
              }
            }
          },
          "physics": {
            "stabilization": {
              "iterations": 100
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": false,
            "tooltipDelay": 100
          }
        }
        """)
        
        # Load the NetworkX graph into pyvis
        nt.from_nx(G)
        
        # Set node colors and sizes based on entity type and connectivity
        for node in nt.nodes:
            # Set node color based on entity type
            entity_type = node.get('group', 'default')
            color = entity_type_colors.get(entity_type, entity_type_colors['default'])
            node['color'] = color
            
            # Adjust node size based on degree
            neighbors = list(G.neighbors(node['id']))
            node['size'] = min(8 + 0.2 * len(neighbors), 16)  # Increased base size
            
            # Set label to be the same as the node ID
            node['label'] = str(node['id'])
            
            # Set node title/tooltip to show more info on hover
            node['title'] = f"{node['id']} | <{entity_type}>"
            
            # Add font configuration to individual nodes
            node['font'] = {'size': 16, 'face': 'arial', 'color': 'black'}
        
        # Customize edge appearance
        for edge in nt.edges:
            source, target = edge['from'], edge['to']
            edge_data = G.get_edge_data(source, target)
            
            if edge_data.get('edge_type') == 'contextual_proximity':
                edge['color'] = '#808080'  # Gray for contextual proximity
                edge['width'] = 1 + edge_data.get('weight', 1) * 0.3  # Thinner for contextual
                edge['dashes'] = True  # Dashed lines for contextual proximity
                # No labels for contextual proximity edges
            else:
                edge['color'] = '#22dd22'  # Green for direct relations
                edge['width'] = 2 + edge_data.get('weight', 1) * 0.5  # Thicker for relations
                
                # Add edge label (only for relation edges, not contextual proximity)
                if edge_data.get('title'):
                    edge['label'] = edge_data['title']
                    # Configure edge label font
                    edge['font'] = {'size': 12, 'align': 'middle', 'color': '#252525', 'strokeWidth': 2, 'strokeColor': '#ffffff'}
            
            # Add title/tooltip to show relation name on hover
            if edge_data.get('title'):
                edge['title'] = edge_data['title']
        
        # Generate the HTML
        if progress:
            progress(0.9, desc="Generating HTML")
            
        html = nt.generate_html()
        
        # Fix quotes to ensure proper embedding
        html = html.replace("'", "\"")
        
        # Create iframe for embedding
        iframe_html = f"""<iframe style="width: 100%; height: 700px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; 
        display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
        allow-scripts allow-same-origin allow-popups 
        allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
        allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
        
        # Save to file if output_path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Saved interactive visualization HTML to {output_path}")
        
        if progress:
            progress(1.0, desc="Visualization complete")
        
        return iframe_html
    
    except Exception as e:
        logger.error(f"Error creating interactive visualization: {e}")
        if progress:
            progress(0.0, desc=f"Error creating visualization: {e}")
        return f"<div>Error creating visualization: {e}</div>" 