import os
import re
import logging
import tempfile
import json
from pathlib import Path
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.ollama_client import OllamaClient
from app.utils.openai_client import OpenAIClient
from app.utils.graph_helpers import chunks2df, df2graph, graph2df, add_ctx_prox_edges, save_graph_to_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def process_single_markdown(md_path, ollama_url, model, chunk_size, chunk_overlap, temp_dir, format_type="json", progress=None):
    """Process a single markdown file and return the path to the temp graph file."""
    md_filename = os.path.basename(md_path)
    logger.info(f"Processing markdown: {md_filename}")
    
    # Load single markdown file with reference sections removed
    loader = MarkdownLoader(md_path)
    documents = loader.load()
    logger.info(f"Loaded document from {md_filename}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {md_filename} into {len(chunks)} chunks")
    
    # Convert documents to dataframe
    df = chunks2df(chunks)
    
    # Save chunks to CSV (temporary)
    temp_chunks_path = os.path.join(temp_dir, f"chunks_{md_filename}.csv")
    df.to_csv(temp_chunks_path, index=False)
    
    # Initialize LLM client
    llm_client = OllamaClient(host=ollama_url)
    # llm_client = OpenAIClient()
    
    # Generate graph from dataframe
    logger.info(f"Generating knowledge graph for {md_filename} using {format_type} format...")
    graph_list = df2graph(df, llm_client, model, format_type=format_type, batch_size=5, progress=progress)
    
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

def markdown_to_knowledge_graph(input_dir, output_file, ollama_url="http://localhost:11434", 
                               model="gemma3:12b", chunk_size=1500, chunk_overlap=200, 
                               format_type="json", progress=None):
    """
    Process all markdown files in the input directory to create a knowledge graph
    
    Args:
        input_dir: Directory containing markdown files
        output_file: Path to save the output knowledge graph
        ollama_url: URL of the Ollama server
        model: Name of the Ollama model to use
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between text chunks
        format_type: Output format type ("json" or "xml")
        progress: Optional progress callback for Gradio
    
    Returns:
        str: Path to the generated knowledge graph file
    """
    # Get all markdown files from input directory
    md_files = list(Path(input_dir).glob('**/*.md'))
    
    if not md_files:
        logger.warning(f"No markdown files found in {input_dir}")
        return None
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Processing {len(md_files)} markdown files...")
        
        # Process each markdown file
        all_graph_paths = []
        for i, md_file in enumerate(md_files):
            if progress:
                # Calculate progress from 0 to 0.2
                current_progress = 0.2 * (i / len(md_files))
                progress(current_progress, desc=f"Processing markdown files: {i+1}/{len(md_files)}")
                
            # Process the file
            graph_path = process_single_markdown(
                md_file, ollama_url, model, chunk_size, chunk_overlap, temp_dir, 
                format_type=format_type, progress=progress
            )
            
            if graph_path:
                all_graph_paths.append(graph_path)
        
        # Combine all graphs
        if progress:
            progress(0.9, desc="Combining knowledge graphs")
            
        if not all_graph_paths:
            logger.error("No valid knowledge graphs generated")
            return None
        
        # Read and combine all graph CSVs
        all_graphs = []
        for graph_path in all_graph_paths:
            try:
                df = pd.read_csv(graph_path)
                all_graphs.append(df)
            except Exception as e:
                logger.error(f"Error reading graph CSV {graph_path}: {e}")
        
        if not all_graphs:
            logger.error("No valid graph CSVs could be read")
            return None
        
        # Combine all graphs into one
        combined_graph = pd.concat(all_graphs, ignore_index=True)
        
        # Save the final graph
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_graph_to_json(combined_graph, output_file)
        
        if progress:
            progress(1.0, desc="Knowledge graph generation complete")
        
        logger.info(f"Knowledge graph saved to {output_file}")
        return output_file 