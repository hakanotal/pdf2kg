from .prompts import generate_graph
from tqdm import tqdm

import pandas as pd
import numpy as np
import logging
import uuid

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


def df2graph(dataframe: pd.DataFrame, model=None, batch_size=5) -> list:
    """Generate knowledge graph from text in dataframe.
    
    Args:
        dataframe: DataFrame containing text columns
        model: Name of the Ollama model to use
        batch_size: Number of rows to process at once with progress updates
        
    Returns:
        List of graph edge dictionaries
    """
    logger.info(f"Generating knowledge graph from {len(dataframe)} text chunks")
    
    all_edges = []
    for i in tqdm(range(0, len(dataframe), batch_size), desc="Generating graph"):
        batch = dataframe.iloc[i:i+batch_size]
        
        # Process each row in the batch
        batch_edges = []
        for _, row in batch.iterrows():
            try:
                edges = generate_graph(row.text, {"chunk_id": row.chunk_id}, model)
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
    graph_dataframe = graph_dataframe[["node_1", "node_1_type", "node_2", "node_2_type", "edge", "chunk_id"]]
    graph_dataframe.dropna(subset=["node_1", "node_1_type", "node_2", "node_2_type"], inplace=True)
    graph_dataframe["count"] = 4 
    graph_dataframe["edge_type"] = "relation"

    for index, row in graph_dataframe.iterrows():
        if row["node_1_type"] not in ["object", "entity", "location", "organization", "person", "condition", "documents", "service", "concept", "date"]:
            graph_dataframe.at[index, "node_1_type"] = "other"

        if row["node_2_type"] not in ["object", "entity", "location", "organization", "person", "condition", "documents", "service", "concept", "date"]:
            graph_dataframe.at[index, "node_2_type"] = "other"
    
    # Normalize text
    for col in ["node_1", "node_2", "node_1_type", "node_2_type"]:
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
    ## Get unique nodes and their types
    df_node_types = df[["node_1","node_1_type"]].rename(columns={"node_1": "node", "node_1_type": "node_type"})
    df_node_types = df_node_types._append(df[["node_2","node_2_type"]].rename(columns={"node_2": "node", "node_2_type": "node_type"}))
    df_node_types.drop_duplicates(subset=["node"], inplace=True)

    ## Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )

    dfg_long.drop(columns=["variable"], inplace=True)

    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))

    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    df_cp = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)

    ## Group and count edges.
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
    df_cp["edge"] = "exists is same context"
    df_cp["node_1_type"] = df_cp["node_1"].apply(lambda x: df_node_types[df_node_types["node"] == x]["node_type"].iloc[0])
    df_cp["node_2_type"] = df_cp["node_2"].apply(lambda x: df_node_types[df_node_types["node"] == x]["node_type"].iloc[0])

    # Combine the two dataframes
    graph_df = (
        pd.concat([df, df_cp], axis=0)
            .groupby(["node_1", "node_2", "edge_type"])
            .agg({
                "node_1_type": "first", 
                "node_2_type": "first", 
                "edge": ",".join, 
                "count": "sum", 
                "chunk_id": ",".join, 
            }).reset_index()
    )

    return graph_df



# def df2ConceptsList(dataframe: pd.DataFrame, model=None, batch_size=10) -> list:
#     """Extract concepts from text in dataframe.
    
#     Args:
#         dataframe: DataFrame containing text columns
#         model: Name of the Ollama model to use
#         batch_size: Number of rows to process at once with progress updates
        
#     Returns:
#         List of concept dictionaries
#     """
#     logger.info(f"Extracting concepts from {len(dataframe)} text chunks")
    
#     all_results = []
#     for i in tqdm(range(0, len(dataframe), batch_size), desc="Extracting concepts"):
#         batch = dataframe.iloc[i:i+batch_size]
        
#         # Process each row in the batch
#         batch_results = []
#         for _, row in batch.iterrows():
#             concepts = extractConcepts(
#                 row.text, {"chunk_id": row.chunk_id, "type": "concept"}, model
#             )
#             if concepts:
#                 batch_results.append(concepts)
        
#         # Flatten and add to all results
#         for result in batch_results:
#             if result:
#                 all_results.extend(result)
                
#         # Log progress
#         logger.info(f"Processed {min(i+batch_size, len(dataframe))}/{len(dataframe)} chunks")
    
#     return all_results


# def concepts2Df(concepts_list) -> pd.DataFrame:
#     """Convert concepts list to a DataFrame.
    
#     Args:
#         concepts_list: List of concept dictionaries
        
#     Returns:
#         DataFrame of concepts with cleaned values
#     """
#     if not concepts_list:
#         logger.warning("Empty concepts list provided")
#         return pd.DataFrame()
        
#     logger.info(f"Converting {len(concepts_list)} concepts to DataFrame")
    
#     # Create DataFrame
#     concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    
#     # Clean up data
#     concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
#     concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
#         lambda x: str(x).lower().strip()
#     )

#     return concepts_dataframe
