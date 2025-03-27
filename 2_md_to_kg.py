from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.helpers import chunks2df, df2graph, graph2df, add_ctx_prox_edges
from tqdm import tqdm

import pandas as pd
import datetime
import logging
import argparse
import uuid
import time
import os
import glob
import re
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def clean_markdown_text(file_path):
    """Remove references section from markdown file and return cleaned text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match common references section headers (case insensitive)
    reference_patterns = [
        r'#{1,6}\s*references\s*$',
        r'#{1,6}\s*bibliography\s*$',
    ]
    
    # Check for each reference pattern
    for pattern in reference_patterns:
        # Look for the pattern in the content
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            # Find the position of the reference section
            ref_start_pos = match.start()
            
            # Look for the next heading at the same or higher level after references
            heading_level = content[ref_start_pos:ref_start_pos+10].count('#')
            next_heading_pattern = r'#{1,' + str(heading_level) + r'}\s+\w+'
            next_heading = re.search(next_heading_pattern, content[ref_start_pos+1:])
            
            if next_heading:
                # Cut content between references and next heading
                clean_content = content[:ref_start_pos] + content[ref_start_pos + next_heading.start() + 1:]
            else:
                # No next heading found, cut everything after references
                clean_content = content[:ref_start_pos]
            
            logger.info(f"Removed references section from {os.path.basename(file_path)}")
            return clean_content
    
    # No reference section found
    return content

class MarkdownLoader:
    """Custom loader for markdown files that removes reference sections."""
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        # Clean the markdown text by removing references sections
        cleaned_text = clean_markdown_text(self.file_path)
        
        # Write cleaned text to a temporary file
        temp_file = f"{self.file_path}.temp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Use TextLoader to load the cleaned file
        loader = TextLoader(temp_file)
        documents = loader.load()
        
        # Remove the temporary file
        try:
            os.remove(temp_file)
        except Exception as e:
            logger.warning(f"Could not remove temp file {temp_file}: {e}")
        
        return documents

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate knowledge graph from markdown files')
    parser.add_argument('--input_dir', type=str, default='data_input/osint_md',
                        help='Directory containing markdown files')
    parser.add_argument('--output_file', type=str, default='data_output/osint_small/knowledge_graph.json',
                        help='Path to save the output knowledge graph')
    parser.add_argument('--model', type=str, default='gemma3:12b',
                        help='Ollama model to use for knowledge graph generation')
    parser.add_argument('--chunk-size', type=int, default=1500,
                        help='Size of text chunks for processing')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                        help='Overlap between text chunks')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of chunks to process in each batch')
    return parser.parse_args()

def process_single_markdown(md_path, args, temp_dir):
    """Process a single markdown file and return the path to the temp graph file."""
    md_filename = os.path.basename(md_path)
    logger.info(f"Processing markdown: {md_filename}")
    
    # Load single markdown file with reference sections removed
    loader = MarkdownLoader(md_path)
    documents = loader.load()
    logger.info(f"Loaded document from {md_filename}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {md_filename} into {len(chunks)} chunks")
    
    # Convert documents to dataframe
    df = chunks2df(chunks)
    
    # Save chunks to CSV (temporary)
    temp_chunks_path = os.path.join(temp_dir, f"chunks_{md_filename}.csv")
    df.to_csv(temp_chunks_path, index=False)
    
    # Generate graph from dataframe
    logger.info(f"Generating knowledge graph for {md_filename}...")
    graph_list = df2graph(df, args.model, args.batch_size)
    
    if not graph_list:
        logger.error(f"Failed to generate knowledge graph edges for {md_filename}")
        return None
    
    # Convert graph list to dataframe
    graph_df_raw = graph2df(graph_list)
    graph_df = add_ctx_prox_edges(graph_df_raw)
    
    # Add source information
    graph_df['source_md'] = md_filename
    
    # Save graph to CSV (temporary)
    temp_graph_path = os.path.join(temp_dir, f"graph_{md_filename}.csv")
    graph_df.to_csv(temp_graph_path, index=False)
    logger.info(f"Saved temporary knowledge graph for {md_filename}")
    
    return temp_graph_path

def process_markdown_file(file_path):
    """
    Process a single markdown file to extract knowledge graph elements
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        dict: Knowledge graph elements extracted from the file
    """
    # Implementation depends on your KG creation logic
    # This is a placeholder
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract entities, relationships, etc. from markdown
    # This is where your actual extraction logic would go
    
    # For demonstration, just creating a simple placeholder structure
    kg_elements = {
        "source": str(file_path),
        "entities": [],
        "relationships": []
    }
    
    # Your actual extraction logic here
    
    return kg_elements

def markdown_to_knowledge_graph(input_dir, output_file):
    """
    Process all markdown files in the input directory to create a knowledge graph
    
    Args:
        input_dir: Directory containing markdown files
        output_file: Path to save the output knowledge graph
    """
    # Get all markdown files from input directory
    md_files = list(Path(input_dir).glob('**/*.md'))
    
    # Process each file and build the knowledge graph
    knowledge_graph = {
        "metadata": {
            "source_directory": input_dir,
            "file_count": len(md_files)
        },
        "elements": []
    }
    
    for md_file in md_files:
        print(f"Processing: {md_file}")
        kg_elements = process_markdown_file(md_file)
        knowledge_graph["elements"].append(kg_elements)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the knowledge graph
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_graph, f, indent=2)
    
    print(f"Knowledge graph created and saved to {output_file}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Define input and output directories
    input_directory = args.input_dir
    output_file = args.output_file
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create temporary directory for intermediate files
    temp_dir = os.path.join(os.path.dirname(output_file), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    logger.info(f"Starting knowledge graph generation from {input_directory}")
    logger.info(f"Using model: {args.model}")
    start_time = time.time()
    
    # Find all markdown files in the input directory
    md_files = glob.glob(os.path.join(input_directory, "*.md"))
    logger.info(f"Found {len(md_files)} markdown files to process")
    
    # Process each markdown file individually
    temp_graph_files = []
    for md_path in tqdm(md_files, desc="Processing markdown files"):
        temp_graph_path = process_single_markdown(md_path, args, temp_dir)
        if temp_graph_path:
            temp_graph_files.append(temp_graph_path)
    
    # Aggregate all graphs into one
    logger.info("Aggregating individual knowledge graphs...")
    if not temp_graph_files:
        logger.error("No valid knowledge graphs generated")
        return
    
    # Concatenate all temporary graph dataframes
    all_graphs = []
    for graph_file in temp_graph_files:
        try:
            df = pd.read_csv(graph_file)
            all_graphs.append(df)
        except Exception as e:
            logger.error(f"Error reading {graph_file}: {e}")
    
    combined_graph = pd.concat(all_graphs, ignore_index=True)
    
    # Add timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    
    # Save final knowledge graph
    graph_path = os.path.join(temp_dir, f"knowledge_graph_{timestamp}.csv")
    combined_graph.to_csv(graph_path, index=False)
    logger.info(f"Saved combined knowledge graph to {graph_path}")
    
    # Print statistics
    logger.info(f"Knowledge Graph Statistics:")
    logger.info(f"  - Number of edges: {len(combined_graph)}")
    logger.info(f"  - Number of unique nodes: {len(set(combined_graph['node_1'].tolist() + combined_graph['node_2'].tolist()))}")
    logger.info(f"  - Node types: {combined_graph['node_1_type'].value_counts().to_dict()}")
    
    # Save combined chunks
    all_chunks = []
    chunk_files = glob.glob(os.path.join(temp_dir, "chunks_*.csv"))
    for chunk_file in chunk_files:
        try:
            df = pd.read_csv(chunk_file)
            all_chunks.append(df)
        except Exception as e:
            logger.error(f"Error reading {chunk_file}: {e}")
    
    if all_chunks:
        combined_chunks = pd.concat(all_chunks, ignore_index=True)
        chunks_path = os.path.join(temp_dir, "chunks.csv")
        combined_chunks.to_csv(chunks_path, index=False)
        logger.info(f"Saved combined chunks to {chunks_path}")
    
    # Save statistics to file
    stats = {
        "timestamp": timestamp,
        "num_edges": len(combined_graph),
        "num_nodes": len(set(combined_graph['node_1'].tolist() + combined_graph['node_2'].tolist())),
        "node_types": {k: int(v) for k, v in combined_graph['node_1_type'].value_counts().to_dict().items()},
        "md_files_processed": len(md_files),
        "processing_time_seconds": time.time() - start_time
    }
    stats_path = os.path.join(temp_dir, f"stats_{timestamp}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Clean up temporary files
    logger.info("Cleaning up temporary files...")
    for file in glob.glob(os.path.join(temp_dir, "*")):
        try:
            os.remove(file)
        except Exception as e:
            logger.error(f"Error removing {file}: {e}")
    
    try:
        os.rmdir(temp_dir)
        logger.info("Removed temporary directory")
    except Exception as e:
        logger.error(f"Error removing temp directory: {e}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Knowledge graph generation completed in {elapsed_time:.2f} seconds")

    # Save final knowledge graph
    markdown_to_knowledge_graph(input_directory, output_file)

if __name__ == "__main__":
    main()
