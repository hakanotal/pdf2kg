import os
import sys
import yaml
import argparse
from pathlib import Path

# Add the root directory to the path so we can import from the app directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.logger import get_logger
from app.components.filter_small_pdfs import filter_small_pdfs
from app.components.pdf_to_md import convert_pdfs_to_md
from app.components.md_translation import translate_markdown_files
from app.components.md_to_kg import markdown_to_knowledge_graph
from app.components.postprocess_kg import postprocess_knowledge_graph
from app.utils.visualize import create_knowledge_graph_visualization

def load_config(config_path="config.yaml"):
    """Load configuration from the config.yaml file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"✗ Error loading config file: {e}")
        sys.exit(1)

def create_directories(config):
    """Create necessary directories based on configuration."""
    print("📁 Creating directories...")
    
    # Create input directories
    input_dirs = [
        config["directories"]["input"]["all"],
        config["directories"]["input"]["pdf"],
        config["directories"]["input"]["md"]
    ]
    
    # Create output directories
    output_dirs = [
        config["directories"]["output"]["kg"],
        config["directories"]["output"]["final"]
    ]
    
    all_dirs = input_dirs + output_dirs
    
    for dir_path in all_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    return {
        "input_all": os.path.abspath(config["directories"]["input"]["all"]),
        "input_pdf": os.path.abspath(config["directories"]["input"]["pdf"]),
        "input_md": os.path.abspath(config["directories"]["input"]["md"]),
        "output_kg": os.path.abspath(config["directories"]["output"]["kg"]),
        "output_final": os.path.abspath(config["directories"]["output"]["final"])
    }

def run_pipeline(config, args, directories):
    """Run the complete PDF2KG pipeline."""
    logger = get_logger(__name__)
    
    # Use command line arguments or config defaults
    input_dir = args.input_dir or directories["input_all"]
    min_size_kb = args.min_size_kb or config["parameters"]["min_size_kb"]
    ollama_url = args.ollama_url or config["ollama"]["url"]
    vision_model = args.vision_model or config["ollama"]["models"]["vision"]
    kg_model = args.kg_model or config["ollama"]["models"]["kg"]
    chunk_size = args.chunk_size or config["parameters"]["chunk_size"]
    chunk_overlap = args.chunk_overlap or config["parameters"]["chunk_overlap"]
    kg_format = args.kg_format
    
    skip_steps = set()
    if args.skip_steps:
        skip_steps = set(int(x.strip()) for x in args.skip_steps.split(','))
    
    print(f"\n🚀 Starting PDF2KG Pipeline")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {os.path.dirname(directories['output_kg'])}")
    print(f"🔧 Configuration:")
    print(f"  • Min PDF size: {min_size_kb} KB")
    print(f"  • Ollama URL: {ollama_url}")
    print(f"  • Vision model: {vision_model}")
    print(f"  • KG model: {kg_model}")
    print(f"  • Chunk size: {chunk_size}")
    print(f"  • Chunk overlap: {chunk_overlap}")
    print(f"  • KG format: {kg_format}")
    if skip_steps:
        print(f"  • Skipping steps: {sorted(skip_steps)}")
    print()
    
    results = []
    
    try:
        # Step 1: Filter small PDFs
        if 1 not in skip_steps:
            print("1️⃣  Step 1/5: Filtering small PDFs...")
            filtered_count = filter_small_pdfs(input_dir, directories["input_pdf"], min_size_kb)
            results.append(f"✓ Filtered PDFs: {filtered_count} files saved to {directories['input_pdf']}")
            print(f"   ✓ {filtered_count} PDFs filtered and saved")
        else:
            print("1️⃣  Step 1/5: Skipped - Filtering small PDFs")
        
        # Step 2: Convert PDFs to Markdown
        if 2 not in skip_steps:
            print("\n2️⃣  Step 2/5: Converting PDFs to Markdown...")
            md_count = convert_pdfs_to_md(directories["input_pdf"], directories["input_md"], ollama_url, vision_model)
            results.append(f"✓ Converted to Markdown: {md_count} files saved to {directories['input_md']}")
            print(f"   ✓ {md_count} PDFs converted to Markdown")
        else:
            print("\n2️⃣  Step 2/5: Skipped - Converting PDFs to Markdown")

        # Step 3: Translate Markdown files
        if 3 not in skip_steps:
            print("\n3️⃣  Step 3/5: Translating non-English Markdown files...")
            translated_count = translate_markdown_files(directories["input_md"])
            results.append(f"✓ Translated Markdown files: {translated_count} files processed in {directories['input_md']}")
            print(f"   ✓ {translated_count} Markdown files processed for translation")
        else:
            print("\n3️⃣  Step 3/5: Skipped - Translating Markdown files")
        
        # Step 4: Create Knowledge Graph
        if 4 not in skip_steps:
            print("\n4️⃣  Step 4/5: Creating Knowledge Graph...")
            kg_path = os.path.join(directories["output_kg"], f"knowledge_graph.{kg_format}")
            kg_result = markdown_to_knowledge_graph(
                directories["input_md"], kg_path, ollama_url, kg_model, 
                chunk_size, chunk_overlap, kg_format
            )
            results.append(f"✓ Knowledge Graph created: {kg_result} saved to {kg_path}")
            print(f"   ✓ Knowledge Graph saved to {kg_path}")
        else:
            print("\n4️⃣  Step 4/5: Skipped - Creating Knowledge Graph")
        
        # Step 5: Post-process Knowledge Graph
        if 5 not in skip_steps:
            print("\n5️⃣  Step 5/5: Post-processing Knowledge Graph...")
            kg_path = os.path.join(directories["output_kg"], f"knowledge_graph.{kg_format}")
            if os.path.exists(kg_path):
                pp_result, combined_nodes_info = postprocess_knowledge_graph(kg_path, directories["output_final"], config=config)
                results.append(f"✓ Post-processed Knowledge Graph: saved to {directories['output_final']}")
                print(f"   ✓ Post-processed files saved to {directories['output_final']}")
                if combined_nodes_info:
                    print(f"   ℹ️  Redundant nodes eliminated: {combined_nodes_info.count('Combined') if combined_nodes_info else 0}")

                ## Visualize KG
                viz_path = os.path.join(directories["output_final"], "visualization.png")

                create_knowledge_graph_visualization(
                    os.path.join(directories["output_final"], "finalgraph.csv"), 
                    os.path.join(directories["output_final"], "metadata.csv"), 
                    viz_path,
                    figsize=(16, 14),
                    show_contextual_proximity=False
                )
                print(f"   ✓ Static visualization saved to {viz_path}")
            else:
                print(f"   ✗ Knowledge graph file not found: {kg_path}")
                results.append(f"✗ Error: Knowledge graph file not found for post-processing")
        else:
            print("\n5️⃣  Step 5/5: Skipped - Post-processing Knowledge Graph")
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"\n📋 Summary:")
        for result in results:
            print(f"   {result}")
            
        # Show final output files
        final_dir = directories["output_final"]
        if os.path.exists(final_dir):
            final_files = list(Path(final_dir).glob("*"))
            if final_files:
                print(f"\n📄 Final output files in {final_dir}:")
                for file_path in sorted(final_files):
                    print(f"   • {file_path.name}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description='PDF2KG Pipeline Test Script - Run the complete pipeline without UI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py                                    # Run with default config
  python test.py --input-dir ./my_pdfs              # Use custom input directory
  python test.py --kg-format xml --verbose          # Use XML format with verbose output
  python test.py --skip-steps 1,2                   # Skip filtering and PDF conversion steps
  python test.py --min-size-kb 100 --chunk-size 2000  # Override processing parameters
        """
    )
    
    parser.add_argument('--input-dir', type=str, help='Input directory containing PDF files')
    parser.add_argument('--output-dir', type=str, help='Output directory (not used, outputs to config paths)')
    parser.add_argument('--min-size-kb', type=int, help='Minimum PDF size in KB')
    parser.add_argument('--ollama-url', type=str, help='Ollama server URL')
    parser.add_argument('--vision-model', type=str, help='Vision model name for PDF to Markdown')
    parser.add_argument('--kg-model', type=str, help='Knowledge graph model name')
    parser.add_argument('--chunk-size', type=int, help='Chunk size for processing')
    parser.add_argument('--chunk-overlap', type=int, help='Chunk overlap size')
    parser.add_argument('--kg-format', type=str, choices=['json', 'xml'], default='json', 
                       help='Knowledge graph format (default: json)')
    parser.add_argument('--skip-steps', type=str, help='Comma-separated list of steps to skip (1,2,3,4,5)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    directories = create_directories(config)
    
    # Validate input directory
    input_dir = args.input_dir or directories["input_all"]
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        print(f"💡 Please create the directory and add PDF files, or specify a different directory with --input-dir")
        sys.exit(1)
    
    # Check if input directory has PDF files
    # pdf_files = list(Path(input_dir).glob("*.pdf"))
    # if not pdf_files:
    #     print(f"⚠️  No PDF files found in input directory: {input_dir}")
    #     print(f"💡 Please add PDF files to the directory or specify a different directory with --input-dir")
    #     sys.exit(1)
    
    # print(f"📄 Found {len(pdf_files)} PDF files in input directory")
    
    # Run the pipeline
    run_pipeline(config, args, directories)

if __name__ == "__main__":
    main()
