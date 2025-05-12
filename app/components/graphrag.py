import pandas as pd
import networkx as nx
import community as community_louvain  # python-louvain library
from collections import defaultdict
import time
import os
from app.utils.ollama_client import OllamaClient
from app.utils.logger import get_logger

# Get logger
logger = get_logger(__name__)

def load_graph_from_csv(filepath: str) -> nx.Graph:
    """
    Loads a knowledge graph from a CSV file.
    CSV format: "source,target,edge,edge_type,value"
    """
    try:
        df = pd.read_csv(filepath)
        G = nx.Graph()  # Use Graph for undirected, or DiGraph if direction matters

        required_cols = ['source', 'target', 'edge', 'edge_type', 'value']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"CSV missing required columns: {', '.join(required_cols)}")
            raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")

        for _, row in df.iterrows():
            source = str(row['source'])
            target = str(row['target'])
            edge_description = str(row['edge'])
            edge_type = str(row['edge_type'])
            # Attempt to convert value to float, handle potential errors
            try:
                edge_weight = float(row['value'])
            except (ValueError, TypeError):
                logger.warning(f"Could not convert value '{row['value']}' to float for edge ({source}, {target}). Using default weight 1.0.")
                edge_weight = 1.0  # Default weight if conversion fails

            # Add nodes (implicitly added by edges, but good practice)
            if source not in G:
                G.add_node(source)  # Add attributes if available
            if target not in G:
                G.add_node(target)

            # Add edge with attributes from the CSV
            G.add_edge(source, target,
                       description=edge_description,
                       type=edge_type,
                       weight=edge_weight)
        
        logger.info(f"Graph loaded successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G, f"Graph loaded successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}")
        return None, f"Error: File not found at {filepath}"
    except Exception as e:
        logger.error(f"Error loading graph from CSV: {e}")
        return None, f"Error loading graph from CSV: {e}"


def detect_communities(graph: nx.Graph) -> tuple:
    """
    Detects communities using the Louvain algorithm.
    Returns a dictionary mapping node_id to community_id and status message.
    """
    if not graph or graph.number_of_nodes() == 0:
        logger.warning("Graph is empty or None. Cannot detect communities.")
        return {}, "Warning: Graph is empty or None. Cannot detect communities."
    
    logger.info("Detecting communities using Louvain algorithm...")
    status_msg = "Detecting communities using Louvain algorithm..."
    # Compute the best partition using Louvain
    # Weight parameter can be adjusted, using 'weight' from CSV if available
    partition = community_louvain.best_partition(graph, weight='weight', random_state=42)
    num_communities = len(set(partition.values()))
    logger.info(f"Detected {num_communities} communities.")
    status_msg += f"\nDetected {num_communities} communities."
    return partition, status_msg


def score_helpfulness(partial_answer: str) -> int:
    """
    Extract the helpfulness score from a partial answer.
    The score should be embedded in the format '(Score: XX/100)'.
    """
    try:
        # Attempt to extract the score embedded in the response
        score_str = partial_answer.split("(Score: ")[1].split("/")[0]
        score = int(score_str)
        return score
    except (IndexError, ValueError):
        # Fallback score if extraction fails
        logger.warning("Failed to extract helpfulness score from partial answer. Using default score 50.")
        return 50  # Default middle score


def generate_community_summaries(graph: nx.Graph, partition: dict, ollama_client: OllamaClient, model: str) -> tuple:
    """
    Generates summaries for each community using LLM.
    Returns community summaries and status message.
    """
    if not partition:
        logger.warning("No partition data provided. Cannot generate summaries.")
        return {}, "Warning: No partition data provided. Cannot generate summaries."

    logger.info("Generating community summaries...")
    status_msg = "Generating community summaries..."
    community_summaries = {}
    communities = defaultdict(list)
    # Group nodes by community ID
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    # Create a summary for each community
    for comm_id, nodes in communities.items():
        # Gather basic info about the community subgraph
        subgraph = graph.subgraph(nodes)
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()

        # Create context string with some nodes and edges for the LLM
        context_str = f"Community ID: {comm_id}\n"
        context_str += f"Number of Nodes: {num_nodes}\n"
        context_str += f"Number of Edges: {num_edges}\n"
        # Add some node names
        for i, node in enumerate(nodes[:5]):  # Limit to first 5 nodes
             context_str += f"Node: {node}\n"
        # Add some edge details
        edges_added = 0
        for u, v, data in subgraph.edges(data=True):
            if edges_added < 5:  # Limit to first 5 edges
                 context_str += f"Edge: ({u}, {v}), Type: {data.get('type', 'N/A')}, Desc: {data.get('description', 'N/A')[:30]}...\n"
                 edges_added += 1
            else:
                break

        # Use the LLM to generate the summary
        logger.info(f"Generating summary for community {comm_id}")
        prompt = "Generate a comprehensive report of a community based on the provided nodes and edges. Focus on key entities and relationships."
        summary = ollama_client.query_llm(prompt, context=context_str, model=model)
        community_summaries[comm_id] = summary

    logger.info(f"Generated {len(community_summaries)} community summaries.")
    status_msg += f"\nGenerated {len(community_summaries)} community summaries."
    return community_summaries, status_msg


def query_graphrag_engine(community_summaries: dict, query: str, ollama_client: OllamaClient, model: str) -> tuple:
    """
    Processes a query using the GraphRAG map-reduce approach.
    Returns the final answer and detailed process notes.
    """
    if not community_summaries:
        logger.error("No community summaries available to process the query.")
        return "Error: No community summaries available to process the query.", "No community summaries available."

    logger.info(f"Starting Query Processing for: '{query}'")
    process_notes = f"Starting Query Processing for: '{query}'\n"
    process_notes += f"Processing {len(community_summaries)} community summaries.\n"

    # 1. Map Step: Generate partial answers from each community summary
    partial_answers = []
    process_notes += "\n--- Map Step: Generating Partial Answers ---\n"
    for comm_id, summary in community_summaries.items():
        logger.info(f"Generating partial answer from community {comm_id}")
        prompt = f"Based *only* on the following community summary, generate a partial answer to the query: '{query}'. Also include a helpfulness score (0-100) for this partial answer in the format '(Score: SCORE/100)'."
        partial_answer = ollama_client.query_llm(prompt, context=summary, model=model)
        process_notes += f"\nCommunity {comm_id} Partial Answer:\n{partial_answer[:300]}...\n"
        
        if partial_answer:  # Ensure LLM returned something
             score = score_helpfulness(partial_answer)
             partial_answers.append({"id": comm_id, "answer": partial_answer, "score": score})
             process_notes += f"Helpfulness Score: {score}/100\n"
        else:
             logger.warning(f"No partial answer generated for community {comm_id}")
             process_notes += "Warning: No partial answer generated for this community\n"

    if not partial_answers:
        logger.error("Failed to generate any partial answers.")
        return "Error: Failed to generate any partial answers.", process_notes

    # 2. Filter and Sort Partial Answers (Implicit in Reduce)
    # Sort by helpfulness score in descending order
    partial_answers.sort(key=lambda x: x['score'], reverse=True)
    process_notes += f"\n--- Reduce Step: Processing {len(partial_answers)} Partial Answers ---\n"
    process_notes += "Top Partial Answers (by score):\n"
    for i, pa in enumerate(partial_answers[:3]):  # Show top 3 for brevity
        process_notes += f"  {i+1}. Community ID {pa['id']}, Score: {pa['score']}, Answer: {pa['answer'][:100]}...\n"

    # 3. Reduce Step: Combine top N partial answers and generate final answer
    # Determine how many partial answers to combine
    num_answers_to_combine = min(len(partial_answers), 5)  # Combine top 5
    combined_context = "\n---\n".join([pa['answer'] for pa in partial_answers[:num_answers_to_combine]])

    logger.info("Generating final answer by combining top partial answers")
    process_notes += f"\n--- Generating Final Answer ---\n"
    final_prompt = f"Synthesize the following partial answers into a single, comprehensive final global answer for the query: '{query}'. Ensure the final answer is coherent and addresses the query directly, citing community references where appropriate."
    final_answer = ollama_client.query_llm(final_prompt, context=combined_context, model=model)

    logger.info("Query processing complete")
    process_notes += "\n--- Query Processing Complete ---\n"
    return final_answer, process_notes


def process_graph_query(kg_dir: str, query: str, ollama_host: str = "http://localhost:11434", model: str = "gemma3:12b") -> tuple:
    """
    Complete GraphRAG workflow: load graph, detect communities, 
    generate summaries, and process the query.
    
    Returns: (final_answer, process_notes)
    """
    logger.info(f"Starting GraphRAG process for query: '{query}'")
    process_notes = ""
    
    # Initialize OllamaClient
    ollama_client = OllamaClient(host=ollama_host)
    
    # 1. Load the graph
    logger.info(f"Loading knowledge graph from {kg_dir}")
    csv_filepath = os.path.join(kg_dir, "finalgraph.csv")
    graph, status = load_graph_from_csv(csv_filepath)
    process_notes += status + "\n\n"
    
    if not graph:
        logger.error("Failed to load knowledge graph.")
        return "Error: Failed to load knowledge graph.", process_notes
    
    # 2. Detect communities
    partition, status = detect_communities(graph)
    process_notes += status + "\n\n"
    
    if not partition:
        logger.error("Failed to detect communities in the graph.")
        return "Error: Failed to detect communities in the graph.", process_notes
    
    # 3. Generate community summaries
    community_summaries, status = generate_community_summaries(graph, partition, ollama_client, model)
    process_notes += status + "\n\n"
    
    if not community_summaries:
        logger.error("Failed to generate community summaries.")
        return "Error: Failed to generate community summaries.", process_notes
    
    # 4. Process the query
    logger.info("Processing query with generated community summaries")
    final_answer, query_process = query_graphrag_engine(
        community_summaries, query, ollama_client, model)
    process_notes += query_process
    
    logger.info("Graph query processing complete")
    return final_answer, process_notes 