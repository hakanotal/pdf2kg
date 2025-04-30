import os
import subprocess
import sys
import glob
from pathlib import Path
import shutil

def convert_pdfs_to_md(input_dir, output_dir, ollama_url="http://localhost:11434", ollama_model="llama3.2-vision:11b", progress=None):
    """
    Convert PDF files to Markdown format using marker-pdf
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save Markdown files
        ollama_url: URL of the Ollama server
        ollama_model: Name of the Ollama model to use
        progress: Optional progress callback for Gradio
    
    Returns:
        int: Number of converted Markdown files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if progress:
        progress(0.1, desc="Installing marker-pdf package")
    
    # Install marker-pdf if not already installed
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "marker-pdf[full]"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing marker-pdf: {e}")
        print(f"Error output: {e.stderr.decode()}")
        return 0
    
    if progress:
        progress(0.2, desc="Converting PDFs to Markdown")
    
    # Convert PDFs to Markdown using marker
    try:
        cmd = [
            "marker",
            "--workers", "2",
            "--use_llm",
            "--disable_image_extraction",
            "--ollama_base_url", ollama_url,
            "--ollama_model", ollama_model,
            "--llm_service=marker.services.ollama.OllamaService",
            "--languages", "en",
            "--output_format", "markdown",
            input_dir,
            "--output_dir", output_dir
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Monitor the conversion process
        total_files = len(list(Path(input_dir).glob('**/*.pdf')))
        processed_files = 0
        
        for line in process.stdout:
            print(line, end='')
            # Look for lines indicating progress
            if "Processing" in line and ".pdf" in line:
                processed_files += 1
                if progress:
                    # Calculate progress from 20% to 80%
                    current_progress = 0.2 + (0.6 * (processed_files / total_files))
                    progress(current_progress, desc=f"Converting PDFs: {processed_files}/{total_files}")
        
        # Wait for the process to finish
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            print(f"Error converting PDFs: {stderr}")
            return 0
        
    except Exception as e:
        print(f"Error converting PDFs: {e}")
        return 0
    
    if progress:
        progress(0.9, desc="Organizing Markdown files")
    
    # Move all Markdown files to the output directory
    try:
        # Find all .md files in subdirectories and move them to the top level
        for md_file in Path(output_dir).glob('**/*.md'):
            if md_file.parent != Path(output_dir):
                shutil.move(str(md_file), str(Path(output_dir) / md_file.name))
        
        # Remove empty subdirectories
        for dir_path in Path(output_dir).glob('*'):
            if dir_path.is_dir():
                shutil.rmtree(str(dir_path))
    except Exception as e:
        print(f"Error organizing Markdown files: {e}")
    
    # Count the number of Markdown files
    md_files = list(Path(output_dir).glob('*.md'))
    md_count = len(md_files)
    
    if progress:
        progress(1.0, desc=f"Conversion complete: {md_count} Markdown files")
    
    return md_count 