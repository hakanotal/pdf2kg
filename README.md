

# [TITLE]


You can download the paper via: [[PAPER]](https://)


## Abstract


## Citation

If this work is helpful, please cite as:

```bibtex
@misc{
}
```

## Visualization

[Interactive Knowledge Graph Visualization](https://)

## Contact

hotal [AT] albany [DOT] edu



# Project Overview
This project provides a pipeline for converting PDF documents into knowledge graphs (KG). It processes PDFs through multiple stages to extract structured information that can be used for knowledge representation and reasoning.

## Outline
The PDF2KG pipeline consists of the following steps:
Filter out small PDFs that may not contain useful content
Convert PDFs to Markdown format
Process Markdown files to create a knowledge graph
Post-process the knowledge graph for improved quality

### Components
0. Filter Small PDFs (0_filter_small_pdfs.py)

This script filters out PDF documents that are below a certain size threshold, as they likely don't contain enough meaningful content to process.

1. PDF to Markdown Conversion (1_pdf_to_md.sh)

A shell script that converts PDF documents to Markdown format, preserving the text content and basic structure.

2. Markdown to Knowledge Graph (2_md_to_kg.py)

Processes the Markdown files to extract entities, relationships, and other structured information to build a knowledge graph.

3. Knowledge Graph Post-Processing (3_postprocess_kg.py)

Refines the knowledge graph by cleaning data, resolving entity references, removing duplicates, and other optimization steps.

### Scripts
The pdf2kg.py script orchestrates the entire pipeline, allowing you to process PDFs through all stages with a single command.

To process PDFs through the entire pipeline:

0. Setup local Ollama server

- You need to install Ollama on your server and change the model names and server urls according to your setup.

0. Installation
```bash
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

1. Filter small PDFs (optional):
```bash
python 0_filter_small_pdfs.py --input_dir ./data_input/osint_all --output_dir ./data_input/osint_small 
```

2. Convert PDFs to Markdown:
```bash
# converts pdfs in ./data_input/osint_small to mardowns in ./data_input/osint_md
sh 1_pdf_to_md.sh ./data_input/osint_small ./data_input/osint_md
```

3. Generate knowledge graph from Markdown:
```bash
python 2_md_to_kg.py --input_dir ./data_input/osint_md --output_file ./data_output/osint_small
```

4. Post-process the knowledge graph:
```bash
python 3_postprocess_kg.py --kg ./data_output/osint_small --output data_output/osint_small
```


## Requirements
- Python 3.11
- Required Python packages (see requirements.txt)
- External tools for PDF processing
