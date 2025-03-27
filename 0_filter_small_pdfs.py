#!/usr/bin/env python3

import argparse
import os
import shutil
from PyPDF2 import PdfReader
from pathlib import Path

def count_pdf_pages(pdf_path):
    """Count the number of pages in a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            return len(pdf.pages)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return float('inf')  # Return infinite pages on error to avoid moving

def move_small_pdfs(source_dir, target_dir, max_pages=50):
    """Move PDFs with fewer than max_pages from source_dir to target_dir."""
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Track statistics
    total_pdfs = 0
    moved_pdfs = 0
    
    # Walk through all files in source directory
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                total_pdfs += 1
                pdf_path = os.path.join(root, filename)
                
                # Get page count
                page_count = count_pdf_pages(pdf_path)
                
                # Move if under the threshold
                if page_count < max_pages:
                    target_path = os.path.join(target_dir, filename)
                    print(f"Moving {filename} ({page_count} pages) to {target_dir}")
                    shutil.move(pdf_path, target_path)
                    moved_pdfs += 1
                else:
                    print(f"Skipping {filename} ({page_count} pages)")
    
    print(f"\nSummary: Moved {moved_pdfs} of {total_pdfs} PDF files")

def filter_small_pdfs(input_dir, output_dir, min_size_kb=50):
    """
    Filter out PDF files smaller than min_size_kb
    
    Args:
        input_dir: Directory containing PDF files to filter
        output_dir: Directory to save filtered PDF files
        min_size_kb: Minimum file size in KB (default: 50)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files from input directory
    pdf_files = list(Path(input_dir).glob('**/*.pdf'))
    
    # Filter and copy files larger than min_size_kb
    for pdf_file in pdf_files:
        file_size_kb = os.path.getsize(pdf_file) / 1024
        if file_size_kb >= min_size_kb:
            # Preserve directory structure
            relative_path = pdf_file.relative_to(input_dir)
            output_path = Path(output_dir) / relative_path
            
            # Create parent directories if they don't exist
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Copy the file
            shutil.copy2(pdf_file, output_path)
            print(f"Copied: {pdf_file} ({file_size_kb:.2f} KB)")
        else:
            print(f"Skipped: {pdf_file} ({file_size_kb:.2f} KB) - too small")

def main():
    parser = argparse.ArgumentParser(description='Filter out small PDF files.')
    parser.add_argument('--input_dir', required=True, help='Directory containing PDF files to filter')
    parser.add_argument('--output_dir', required=True, help='Directory to save filtered PDF files')
    parser.add_argument('--min_size_kb', type=int, default=50, help='Minimum file size in KB (default: 50)')
    
    args = parser.parse_args()
    
    filter_small_pdfs(args.input_dir, args.output_dir, args.min_size_kb)

if __name__ == "__main__":
    main() 