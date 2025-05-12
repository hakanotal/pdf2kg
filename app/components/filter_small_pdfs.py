import os
import shutil
from PyPDF2 import PdfReader
from pathlib import Path
from tqdm import tqdm
from app.utils.logger import get_logger

# Get logger
logger = get_logger(__name__)

def filter_small_pdfs(input_dir, output_dir, min_size_kb=50, progress=None):
    """
    Filter out PDF files smaller than min_size_kb
    
    Args:
        input_dir: Directory containing PDF files to filter
        output_dir: Directory to save filtered PDF files
        min_size_kb: Minimum file size in KB (default: 50)
        progress: Optional progress callback for Gradio
    
    Returns:
        int: Number of filtered PDF files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files from input directory
    pdf_files = list(Path(input_dir).glob('**/*.pdf'))
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    # Set up progress tracking
    filtered_count = 0
    total_files = len(pdf_files)
    
    # Filter and copy files larger than min_size_kb
    for i, pdf_file in enumerate(pdf_files):
        # Update progress if provided
        if progress is not None:
            progress((i + 1) / total_files, desc=f"Filtering PDFs: {i+1}/{total_files}")
        
        file_size_kb = os.path.getsize(pdf_file) / 1024
        if file_size_kb >= min_size_kb:
            # Check if it's a valid PDF by trying to read it
            try:
                with open(pdf_file, 'rb') as f:
                    PdfReader(f)
                
                # Preserve directory structure
                relative_path = pdf_file.relative_to(input_dir)
                output_path = Path(output_dir) / relative_path
                
                # Create parent directories if they don't exist
                os.makedirs(output_path.parent, exist_ok=True)
                
                # Copy the file
                shutil.copy2(pdf_file, output_path)
                filtered_count += 1
            except Exception as e:
                logger.warning(f"Skipped invalid PDF: {pdf_file} - {str(e)}")
    
    logger.info(f"Filtered {filtered_count} PDF files (size >= {min_size_kb}KB) from {total_files} total files")
    return filtered_count 