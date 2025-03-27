#!/bin/bash

# Check if input and output directories are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"


echo "Installing marker-pdf"
pip install -qq -U marker-pdf[full]


echo "Converting pdfs to markdown"
marker \
    --workers 2 \
    --use_llm \
    --disable_image_extraction \
    --ollama_base_url http://169.226.53.98:11434 \
    --ollama_model llama3.2-vision:11b \
    --llm_service=marker.services.ollama.OllamaService \
    --languages "en" \
    --output_format markdown \
    $INPUT_DIR \
    --output_dir $OUTPUT_DIR


find "$OUTPUT_DIR" -mindepth 2 -type f -name "*.md" -exec mv {} "$OUTPUT_DIR" \;
find "$OUTPUT_DIR" -mindepth 1 -type d -exec rm -rf {} \;
echo "Output directory: $OUTPUT_DIR"