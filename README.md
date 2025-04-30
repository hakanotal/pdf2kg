# PDF2KG: PDF to Knowledge Graph Converter

Convert PDF documents into knowledge graphs for knowledge representation and reasoning.

## Overview

PDF2KG provides a complete pipeline for converting PDF documents into knowledge graphs. It processes PDFs through multiple stages: filtering small PDFs, converting to markdown, generating a knowledge graph, post-processing, and visualization.

## Installation

```bash
# Create a virtual environment and install dependencies
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt

# Install Ollama models
ollama pull llama3.2-vision:11b
ollama pull gemma3:1b
```

## Configuration

PDF2KG uses a `config.yaml` file to customize directories, models, and parameters.


## Usage

### Running the Application (Gradio)

```bash
python main.py [options]
```

Options:
- `--host`: Host address (default: 127.0.0.1)
- `--port`: Port number (default: 7860)
- `--share`: Create a publicly shareable link
- `--auto-port`: Find an available port automatically

### Using the Interface

The application provides a Gradio web interface with these tabs:

1. **Full Pipeline**: Run the entire conversion process in one go
2. **1. Filter PDFs**: Remove PDFs below a size threshold
3. **2. PDF to Markdown**: Transform PDF documents to Markdown
4. **3. Markdown to KG**: Generate a knowledge graph from Markdown
5. **4. Post-process**: Refine the knowledge graph
6. **5. Visualization**: Interactively visualize the knowledge graph

### Workflow

1. Place your PDF files in `input/all/` directory
2. Run the application with `python main.py`
3. Use the "Full Pipeline" tab or process step by step
4. View results in the `output/final/` directory
5. Visualize the knowledge graph in the "Visualization" tab

## Project Structure

```
pdf2kg/
├── app/                   # Main application code
│   ├── components/        # Pipeline components
│   ├── utils/             # Utility functions
│   └── app.py             # Gradio application
├── config.yaml            # Configuration file
├── input/                 # Input directories
├── output/                # Output directories
├── main.py                # Entry point
└── requirements.txt       # Dependencies
```

## Requirements

- Python 3.11+
- Ollama with `llama3.2-vision:11b` and `gemma3:1b` models
- Dependencies listed in requirements.txt

## Contact

For questions or feedback: hotal [AT] albany [DOT] edu
