import pandas as pd
import numpy as np
import uuid
from tqdm import tqdm
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def chunks2df(documents) -> pd.DataFrame:
    """Convert document chunks to a DataFrame.
    
    Args:
        documents: List of document chunks from LangChain document loader
        
    Returns:
        DataFrame with text content and metadata
    """
    rows = []
    logger.info(f"Converting {len(documents)} document chunks to DataFrame")
    
    for chunk in tqdm(documents, desc="Processing document chunks"):
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def df2graph(dataframe: pd.DataFrame, ollama_client, model=None, batch_size=5, progress=None) -> list:
    """Generate knowledge graph from text in dataframe.
    
    Args:
        dataframe: DataFrame containing text columns
        ollama_client: OllamaClient instance
        model: Name of the Ollama model to use
        batch_size: Number of rows to process at once with progress updates
        progress: Optional progress callback for Gradio
        
    Returns:
        List of graph edge dictionaries
    """
    logger.info(f"Generating knowledge graph from {len(dataframe)} text chunks")
    
    all_edges = []
    total_batches = (len(dataframe) + batch_size - 1) // batch_size
    
    for i in range(0, len(dataframe), batch_size):
        if progress:
            # Calculate progress between 0.3 and 0.9
            current_progress = 0.3 + (0.6 * (i / len(dataframe)))
            progress(current_progress, desc=f"Generating graph: batch {i//batch_size+1}/{total_batches}")
            
        batch = dataframe.iloc[i:i+batch_size]
        
        # Process each row in the batch
        batch_edges = []
        for _, row in batch.iterrows():
            try:
                edges = ollama_client.generate_graph(row.text, {"chunk_id": row.chunk_id}, model)
                if edges:
                    batch_edges.append(edges)
            except Exception as e:
                logger.error(f"Error processing chunk {row.chunk_id}: {e}")
        
        # Flatten and add to all edges
        for edges in batch_edges:
            if edges:
                all_edges.extend(edges)
                
        # Log progress
        logger.info(f"Processed {min(i+batch_size, len(dataframe))}/{len(dataframe)} chunks")
    
    return all_edges

def graph2df(nodes_list) -> pd.DataFrame:
    """Convert graph edges list to a DataFrame.
    
    Args:
        nodes_list: List of edge dictionaries
        
    Returns:
        DataFrame of graph edges with cleaned values
    """
    if not nodes_list:
        logger.warning("Empty edges list provided")
        return pd.DataFrame()
        
    logger.info(f"Converting {len(nodes_list)} edges to DataFrame")
    
    # Create DataFrame
    graph_dataframe = pd.DataFrame(nodes_list).replace("", np.nan)
    
    # Clean up data
    graph_dataframe = graph_dataframe[["node_1", "node_2", "edge", "chunk_id"]]
    graph_dataframe.dropna(subset=["node_1", "node_2"], inplace=True)
    graph_dataframe["count"] = 4 
    graph_dataframe["edge_type"] = "relation"
    
    # Normalize text
    for col in ["node_1", "node_2"]:
        if col in graph_dataframe.columns:
            graph_dataframe[col] = graph_dataframe[col].apply(lambda x: str(x).lower().strip() if x is not None else "")
    
    # Remove duplicate edges
    graph_dataframe = graph_dataframe.drop_duplicates(subset=["node_1", "node_2", "edge"])
    
    return graph_dataframe

def add_ctx_prox_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Extract edges from the dataframe based on contextual proximity.

    Args:
        df: DataFrame containing node pairs and their counts

    Returns:
        DataFrame of edges with contextual proximity
    """
    # Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )

    dfg_long.drop(columns=["variable"], inplace=True)

    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))

    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    df_cp = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)

    # Group and count edges.
    df_cp = (
        df_cp.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )

    df_cp.columns = ["node_1", "node_2", "chunk_id", "count"]
    df_cp.replace("", np.nan, inplace=True)
    df_cp.dropna(subset=["node_1", "node_2"], inplace=True)

    # Drop edges with 1 count
    df_cp = df_cp[df_cp["count"] != 1]
    df_cp["edge_type"] = "contextual_proximity"
    df_cp["edge"] = "exists in same context"

    # Combine the two dataframes
    graph_df = (
        pd.concat([df, df_cp], axis=0)
            .groupby(["node_1", "node_2", "edge_type"])
            .agg({
                "edge": ",".join, 
                "count": "sum", 
                "chunk_id": ",".join, 
            }).reset_index()
    )

    return graph_df

def save_graph_to_json(graph_df: pd.DataFrame, output_file: str):
    """
    Save the graph DataFrame to a JSON file with metadata
    
    Args:
        graph_df: DataFrame containing the graph edges
        output_file: Path to save the JSON file
    """
    # Create a dictionary to store the graph data
    graph_data = {
        "metadata": {
            "created_at": pd.Timestamp.now().isoformat(),
            "edge_count": len(graph_df),
            "node_count": len(pd.concat([graph_df['node_1'], graph_df['node_2']]).unique())
        },
        "edges": []
    }
    
    # Convert each row to a dictionary and add to the edges list
    for _, row in graph_df.iterrows():
        edge = {
            "node_1": row["node_1"],
            "node_2": row["node_2"],
            "edge": row["edge"],
            "edge_type": row["edge_type"],
            "count": int(row["count"]),
            "chunk_id": row["chunk_id"]
        }
        graph_data["edges"].append(edge)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved graph with {len(graph_df)} edges to {output_file}")
    
    return output_file 