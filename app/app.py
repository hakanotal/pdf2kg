import os
import sys
import gradio as gr
import tempfile
import yaml
from pathlib import Path
from app.utils.logger import get_logger

# Get logger
logger = get_logger(__name__)

# Add the root directory to the path so we can import from the utils directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components.filter_small_pdfs import filter_small_pdfs
from app.components.pdf_to_md import convert_pdfs_to_md
from app.components.md_translation import translate_markdown_files
from app.components.md_to_kg import markdown_to_knowledge_graph
from app.components.postprocess_kg import postprocess_knowledge_graph
from app.components.graphrag import process_graph_query
from app.utils.visualize import create_knowledge_graph_visualization

def load_config():
    """Load configuration from the config.yaml file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}, using default configuration")
        return {}

# Create default directories
def create_default_directories(config):
    """Create default input and output directories if they don't exist."""
    # Create input directories
    os.makedirs(config["directories"]["input"]["all"], exist_ok=True)
    os.makedirs(config["directories"]["input"]["pdf"], exist_ok=True)
    os.makedirs(config["directories"]["input"]["md"], exist_ok=True)
    
    # Create output directories
    os.makedirs(config["directories"]["output"]["kg"], exist_ok=True)
    os.makedirs(config["directories"]["output"]["final"], exist_ok=True)
    
    logger.info("Default directories created")
    
    return {
        "input_all": os.path.abspath(config["directories"]["input"]["all"]),
        "input_pdf": os.path.abspath(config["directories"]["input"]["pdf"]),
        "input_md": os.path.abspath(config["directories"]["input"]["md"]),
        "output_kg": os.path.abspath(config["directories"]["output"]["kg"]),
        "output_final": os.path.abspath(config["directories"]["output"]["final"])
    }

# Create the main application
def create_app():
    # Load configuration
    config = load_config()
    
    # Create default directories and get their absolute paths
    default_dirs = create_default_directories(config)
    
    with gr.Blocks(title="PDF2KG - PDF to Knowledge Graph Converter", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # PDF2KG
        ## Convert PDF documents into knowledge graphs
        
        This application processes PDFs through multiple stages to extract structured information 
        that can be used for knowledge representation and reasoning.
        """)
        
        with gr.Tabs():
            # Pipeline Tab
            with gr.TabItem("Full Pipeline"):
                with gr.Row():
                    with gr.Column():
                        input_dir = gr.Textbox(label="Input PDF Directory", value=default_dirs["input_all"], placeholder="Path to directory containing PDF files")
                        output_dir = gr.Textbox(label="Output Directory", value=os.path.dirname(default_dirs["output_kg"]), placeholder="Path to save the output files")
                        min_size_kb = gr.Slider(label="Minimum PDF Size (KB)", minimum=0, maximum=1000, value=config["parameters"]["min_size_kb"], step=10)
                        ollama_url = gr.Textbox(label="Ollama Server URL", value=config["ollama"]["url"], placeholder="http://localhost:11434")
                        vision_model = gr.Textbox(label="Vision Model for PDF to Markdown", value=config["ollama"]["models"]["vision"], placeholder="llama3.2-vision:11b")
                        kg_model = gr.Textbox(label="Language Model for Knowledge Graph", value=config["ollama"]["models"]["kg"], placeholder="gemma3:1b")
                        chunk_size = gr.Slider(label="Chunk Size", minimum=500, maximum=3000, value=config["parameters"]["chunk_size"], step=100)
                        chunk_overlap = gr.Slider(label="Chunk Overlap", minimum=0, maximum=500, value=config["parameters"]["chunk_overlap"], step=50)
                        kg_format_type = gr.Dropdown(
                            label="Knowledge Graph Format", 
                            choices=["json", "xml"], 
                            value="json",
                            info="Select the format for knowledge graph extraction"
                        )
                        
                    with gr.Column():
                        status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        run_button = gr.Button("Run Complete Pipeline", variant="primary")
                        output_text = gr.Textbox(label="Output", interactive=False)
                        
                        def run_pipeline(input_dir, output_dir, min_size_kb, ollama_url, vision_model, kg_model, chunk_size, chunk_overlap, kg_format):
                            if not input_dir or not output_dir:
                                return "Please provide both input and output directories."
                            
                            # Create output directories if they don't exist
                            pdf_filtered_dir = default_dirs["input_pdf"]
                            md_dir = default_dirs["input_md"]
                            kg_dir = default_dirs["output_kg"]
                            final_dir = default_dirs["output_final"]
                            
                            results = []
                            
                            # Step 1: Filter small PDFs
                            logger.info("Step 1/5: Filtering small PDFs")
                            filtered_count = filter_small_pdfs(input_dir, pdf_filtered_dir, min_size_kb)
                            results.append(f"Filtered PDFs: {filtered_count} files saved to {pdf_filtered_dir}")
                            
                            # Step 2: Convert PDFs to Markdown
                            logger.info("Step 2/5: Converting PDFs to Markdown")
                            md_count = convert_pdfs_to_md(pdf_filtered_dir, md_dir, ollama_url, vision_model)
                            results.append(f"Converted to Markdown: {md_count} files saved to {md_dir}")
                            
                            # Step 3: Translate Markdown files if needed
                            logger.info("Step 3/5: Translating non-English Markdown files")
                            translated_count = translate_markdown_files(md_dir)
                            results.append(f"Translated Markdown files: {translated_count} files processed in {md_dir}")
                            
                            # Step 4: Create Knowledge Graph
                            logger.info("Step 4/5: Creating Knowledge Graph")
                            kg_path = os.path.join(kg_dir, "knowledge_graph.json")
                            kg_result = markdown_to_knowledge_graph(md_dir, kg_path, ollama_url, kg_model, 
                                                                   chunk_size, chunk_overlap, kg_format)
                            results.append(f"Knowledge Graph created: {kg_result} saved to {kg_path}")
                            
                            # Step 5: Post-process Knowledge Graph
                            logger.info("Step 5/5: Post-processing Knowledge Graph")
                            pp_result, combined_nodes_info = postprocess_knowledge_graph(kg_path, final_dir, config=config)
                            results.append(f"Post-processed Knowledge Graph: saved to {final_dir}")
                            if combined_nodes_info:
                                results.append("\n" + combined_nodes_info)
                            
                            logger.info("Pipeline completed")
                            return "\n".join(results)
                        
                        run_button.click(
                            fn=run_pipeline,
                            inputs=[input_dir, output_dir, min_size_kb, ollama_url, vision_model, 
                                   kg_model, chunk_size, chunk_overlap, kg_format_type],
                            outputs=[output_text]
                        )
            
            # Individual Steps Tabs
            with gr.TabItem("1. Filter PDFs"):
                with gr.Row():
                    with gr.Column():
                        s1_input_dir = gr.Textbox(label="Input PDF Directory", value=default_dirs["input_all"], placeholder="Path to directory containing PDF files")
                        s1_output_dir = gr.Textbox(label="Output Directory", value=default_dirs["input_pdf"], placeholder="Path to save filtered PDFs")
                        s1_min_size_kb = gr.Slider(label="Minimum PDF Size (KB)", minimum=0, maximum=1000, value=config["parameters"]["min_size_kb"], step=10)
                    with gr.Column():
                        s1_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        s1_run_button = gr.Button("Run Filtering", variant="primary")
                        s1_output_text = gr.Textbox(label="Output", interactive=False)
                        
                        def run_step1(input_dir, output_dir, min_size_kb):
                            if not input_dir or not output_dir:
                                return "Please provide both input and output directories."
                            os.makedirs(output_dir, exist_ok=True)
                            filtered_count = filter_small_pdfs(input_dir, output_dir, min_size_kb)
                            return f"Filtered PDFs: {filtered_count} files saved to {output_dir}"
                        
                        s1_run_button.click(
                            fn=run_step1,
                            inputs=[s1_input_dir, s1_output_dir, s1_min_size_kb],
                            outputs=[s1_output_text]
                        )
            
            with gr.TabItem("2. PDF to Markdown"):
                with gr.Row():
                    with gr.Column():
                        s2_input_dir = gr.Textbox(label="Input PDF Directory", value=default_dirs["input_pdf"], placeholder="Path to directory containing PDF files")
                        s2_output_dir = gr.Textbox(label="Output Directory", value=default_dirs["input_md"], placeholder="Path to save Markdown files")
                        s2_ollama_url = gr.Textbox(label="Ollama Server URL", value=config["ollama"]["url"], placeholder="http://localhost:11434")
                        s2_ollama_model = gr.Textbox(label="Ollama Model Name", value=config["ollama"]["models"]["vision"], placeholder="llama3.2-vision:11b")
                    with gr.Column():
                        s2_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        s2_run_button = gr.Button("Convert to Markdown", variant="primary")
                        s2_output_text = gr.Textbox(label="Output", interactive=False)
                        
                        def run_step2(input_dir, output_dir, ollama_url, ollama_model):
                            if not input_dir or not output_dir:
                                return "Please provide both input and output directories."
                            os.makedirs(output_dir, exist_ok=True)
                            md_count = convert_pdfs_to_md(input_dir, output_dir, ollama_url, ollama_model)
                            return f"Converted to Markdown: {md_count} files saved to {output_dir}"
                        
                        s2_run_button.click(
                            fn=run_step2,
                            inputs=[s2_input_dir, s2_output_dir, s2_ollama_url, s2_ollama_model],
                            outputs=[s2_output_text]
                        )
            
            with gr.TabItem("2.5. Translate Markdown"):
                with gr.Row():
                    with gr.Column():
                        s25_input_dir = gr.Textbox(label="Input Markdown Directory", value=default_dirs["input_md"], placeholder="Path to directory containing Markdown files")
                        s25_output_dir = gr.Textbox(label="Output Directory", value=default_dirs["input_md"], placeholder="Path to save translated Markdown files (same as input to overwrite)")
                    with gr.Column():
                        s25_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        s25_run_button = gr.Button("Translate Markdown", variant="primary")
                        s25_output_text = gr.Textbox(label="Output", interactive=False)
                        
                        def run_step25(input_dir, output_dir):
                            if not input_dir:
                                return "Please provide input directory."
                            if not output_dir:
                                output_dir = input_dir  # Default to input dir if not specified (overwrites files)
                            else:
                                os.makedirs(output_dir, exist_ok=True)
                            
                            translated_count = translate_markdown_files(input_dir, output_dir)
                            return f"Translated Markdown files: {translated_count} files processed."
                        
                        s25_run_button.click(
                            fn=run_step25,
                            inputs=[s25_input_dir, s25_output_dir],
                            outputs=[s25_output_text]
                        )
            
            with gr.TabItem("3. Markdown to KG"):
                with gr.Row():
                    with gr.Column():
                        s3_input_dir = gr.Textbox(label="Input Markdown Directory", value=default_dirs["input_md"], placeholder="Path to directory containing Markdown files")
                        s3_output_file = gr.Textbox(label="Output File", value=os.path.join(default_dirs["output_kg"], "knowledge_graph.json"), placeholder="Path to save the knowledge graph")
                        s3_ollama_url = gr.Textbox(label="Ollama Server URL", value=config["ollama"]["url"], placeholder="http://localhost:11434")
                        s3_ollama_model = gr.Textbox(label="Ollama Model Name", value=config["ollama"]["models"]["kg"], placeholder="gemma3:1b")
                        s3_chunk_size = gr.Slider(label="Chunk Size", minimum=500, maximum=3000, value=config["parameters"]["chunk_size"], step=100)
                        s3_chunk_overlap = gr.Slider(label="Chunk Overlap", minimum=0, maximum=500, value=config["parameters"]["chunk_overlap"], step=50)
                        s3_format_type = gr.Dropdown(
                            label="KG Format", 
                            choices=["json", "xml"], 
                            value="json",
                            info="Select the format for knowledge graph extraction"
                        )
                    with gr.Column():
                        s3_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        s3_run_button = gr.Button("Create Knowledge Graph", variant="primary")
                        s3_output_text = gr.Textbox(label="Output", interactive=False)
                        
                        def run_step3(input_dir, output_file, ollama_url, ollama_model, chunk_size, chunk_overlap, format_type):
                            if not input_dir or not output_file:
                                return "Please provide both input directory and output file."
                            os.makedirs(os.path.dirname(output_file), exist_ok=True)
                            kg_result = markdown_to_knowledge_graph(
                                input_dir, output_file, ollama_url, ollama_model, 
                                chunk_size, chunk_overlap, format_type
                            )
                            return f"Knowledge Graph created: {kg_result} saved to {output_file}"
                        
                        s3_run_button.click(
                            fn=run_step3,
                            inputs=[s3_input_dir, s3_output_file, s3_ollama_url, s3_ollama_model, 
                                   s3_chunk_size, s3_chunk_overlap, s3_format_type],
                            outputs=[s3_output_text]
                        )
            
            with gr.TabItem("4. Post-process"):
                with gr.Row():
                    with gr.Column():
                        s4_input_file = gr.Textbox(label="Input Knowledge Graph", value=os.path.join(default_dirs["output_kg"], "knowledge_graph.json"), placeholder="Path to the knowledge graph file")
                        s4_output_path = gr.Textbox(label="Output Path", value=default_dirs["output_final"], placeholder="Path to save the processed knowledge graph")
                    with gr.Column():
                        s4_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        s4_run_button = gr.Button("Post-process Knowledge Graph", variant="primary")
                        s4_output_text = gr.Textbox(label="Output", interactive=False)
                        s4_combined_nodes_info = gr.Textbox(label="Redundant Nodes Eliminated", interactive=False, lines=10)
                        
                        def run_step4(input_file, output_path):
                            if not input_file or not output_path:
                                return "Please provide both input file and output path.", ""
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            output_dir, combined_nodes_info = postprocess_knowledge_graph(input_file, output_path, config=config)
                            return f"Post-processed Knowledge Graph: saved to {output_path}", combined_nodes_info or "No redundant nodes were eliminated."
                        
                        s4_run_button.click(
                            fn=run_step4,
                            inputs=[s4_input_file, s4_output_path],
                            outputs=[s4_output_text, s4_combined_nodes_info]
                        )
            
            # Visualization Tab
            with gr.TabItem("5. Visualization"):
                with gr.Row():
                    with gr.Column(scale=3):
                        viz_kg_dir = gr.Textbox(label="Knowledge Graph Directory", value=default_dirs["output_final"], placeholder="Path to directory containing final_kg files")
                    with gr.Column(scale=1):
                        viz_show_contextual = gr.Checkbox(label="Show Contextual Proximity Edges", value=True, info="Toggle to show/hide contextual proximity edges")
                        viz_run_button = gr.Button("Visualize Knowledge Graph", variant="primary")
                
                with gr.Row():
                    viz_output = gr.Image(label="Knowledge Graph Visualization", interactive=False)
                    
                    def visualize_kg(kg_dir, show_contextual_proximity):
                        if not kg_dir or not os.path.exists(kg_dir):
                            return None
                        
                        # Look for finalgraph.csv
                        finalgraph_path = os.path.join(kg_dir, "finalgraph.csv")
                        metadata_path = os.path.join(kg_dir, "metadata.csv")
                        
                        if not os.path.exists(finalgraph_path):
                            return None
                        
                        # Create visualization
                        viz_path = os.path.join(kg_dir, "visualization.png")
                        
                        result = create_knowledge_graph_visualization(
                            finalgraph_path, 
                            metadata_path if os.path.exists(metadata_path) else None,
                            viz_path,
                            figsize=(12, 10),
                            show_contextual_proximity=show_contextual_proximity
                        )
                        
                        if result and os.path.exists(viz_path):
                            return viz_path
                        else:
                            return None
                    
                    viz_run_button.click(
                        fn=visualize_kg,
                        inputs=[viz_kg_dir, viz_show_contextual],
                        outputs=[viz_output]
                    )
            
            # GraphRAG Tab
            with gr.TabItem("6. GraphRAG Chat"):
                gr.Markdown("""
                # Knowledge Graph Question Answering
                
                Ask questions about your knowledge graph using the GraphRAG approach. 
                The system will analyze communities within the graph and generate answers based on the graph structure.
                """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        rag_kg_dir = gr.Textbox(label="Knowledge Graph Directory", value=default_dirs["output_final"], 
                                             placeholder="Path to directory containing final_kg files")
                        rag_ollama_url = gr.Textbox(label="Ollama Server URL", value=config["ollama"]["url"], 
                                                 placeholder="http://localhost:11434")
                        rag_ollama_model = gr.Textbox(label="Ollama Model Name", value=config["ollama"]["models"]["kg"], 
                                                   placeholder="gemma3:12b")
                        rag_query = gr.Textbox(label="Your Question", placeholder="Ask a question about the knowledge graph...", 
                                            lines=2)
                        rag_run_button = gr.Button("Ask Question", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        rag_process = gr.Textbox(label="Process Notes", interactive=False, lines=15)
                        rag_answer = gr.Textbox(label="Final Answer", interactive=False, lines=10)
                    
                    def run_graphrag(kg_dir, query, ollama_url, model):
                        if not query or not kg_dir:
                            return "Please provide both a question and a knowledge graph directory.", "No query or knowledge graph specified."
                        
                        # Process the query using GraphRAG
                        final_answer, process_notes = process_graph_query(kg_dir, query, ollama_url, model)
                        return process_notes, final_answer
                    
                    rag_run_button.click(
                        fn=run_graphrag,
                        inputs=[rag_kg_dir, rag_query, rag_ollama_url, rag_ollama_model],
                        outputs=[rag_process, rag_answer]
                    )
    
    return app

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    app = create_app()
    app.launch(share=config["ui"]["share"], pwa=config["ui"]["pwa"]) 