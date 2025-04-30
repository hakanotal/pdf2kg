import os
import json
import pandas as pd
import networkx as nx
from pyvis.network import Network
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_knowledge_graph_visualization(graph_data_path, metadata_path=None, output_path="visualization.html", 
                                        height="700px", width="100%", progress=None):
    """
    Create an interactive visualization of the knowledge graph
    
    Args:
        graph_data_path: Path to the graph data CSV (finalgraph.csv)
        metadata_path: Path to the node metadata CSV (metadata.csv)
        output_path: Path to save the HTML visualization
        height: Height of the visualization
        width: Width of the visualization
        progress: Optional progress callback for Gradio
        
    Returns:
        str: Path to the visualization HTML file
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
        
        # Add nodes and edges to the graph
        for _, row in graph_df.iterrows():
            source = row['source']
            target = row['target']
            
            # Add nodes if they don't exist
            if source not in G.nodes:
                G.add_node(source, label=source)
            
            if target not in G.nodes:
                G.add_node(target, label=target)
            
            # Add the edge
            G.add_edge(source, target, 
                      title=row['edge'],
                      weight=row.get('value', 1),
                      color=row.get('color', '#808080'))
        
        # Create the visualization
        if progress:
            progress(0.7, desc="Generating visualization")
        
        # Create a pyvis network
        net = Network(height=height, width=width, notebook=False)
        
        # Set options for a better visualization
        net.set_options('''
        {
          "nodes": {
            "shape": "dot",
            "size": 20,
            "font": {
              "size": 14,
              "face": "Tahoma"
            }
          },
          "edges": {
            "color": {
              "inherit": false
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "maxVelocity": 50,
            "minVelocity": 0.75,
            "solver": "barnesHut"
          }
        }
        ''')
        
        # Add the NetworkX graph to the pyvis network
        net.from_nx(G)
        
        # Save the visualization
        if progress:
            progress(0.9, desc="Saving visualization")
        
        net.save_graph(output_path)
        logger.info(f"Saved visualization to {output_path}")
        
        if progress:
            progress(1.0, desc="Visualization complete")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        if progress:
            progress(0.0, desc=f"Error creating visualization: {e}")
        return None 