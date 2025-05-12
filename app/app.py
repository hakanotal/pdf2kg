import os
import sys
import yaml
import gradio as gr
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
    config = load_config()
    default_dirs = create_default_directories(config)
    
    with gr.Blocks(title="PDF2KG - PDF to Knowledge Graph Converter", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        ## PDF2KG: Convert PDF documents into knowledge graphs
        
        This application processes PDFs through multiple stages to extract structured information 
        that can be used for knowledge representation and reasoning.
        """)
        
        sections = []

        with gr.Row():
            with gr.Sidebar():
                gr.Markdown("# PDF2KG")
                gr.Markdown("------")
                b_pipeline = gr.Button("Full Pipeline")
                gr.Markdown("------")
                b_s1 = gr.Button("1. Filter PDFs")
                b_s2 = gr.Button("2. PDF to Markdown")
                b_s25 = gr.Button("3. Translate Markdown")
                b_s3 = gr.Button("4. Markdown to KG")
                b_s4 = gr.Button("5. Post-process KG")
                gr.Markdown("------")
                b_viz = gr.Button("Visualize KG")
                b_rag = gr.Button("GraphRAG Chat")
                gr.Markdown("------")

            with gr.Column(scale=3): # Main content area
                # Section: Full Pipeline
                with gr.Column(visible=True) as section_pipeline_ui:
                    gr.Markdown("### Full Pipeline")
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
                            output_text = gr.Textbox(label="Output", interactive=False, lines=10, max_lines=20)
                            
                            def run_pipeline(input_dir_val, output_dir_val, min_size_kb_val, ollama_url_val, vision_model_val, kg_model_val, chunk_size_val, chunk_overlap_val, kg_format_val):
                                if not input_dir_val or not output_dir_val:
                                    return "Please provide both input and output directories."
                                
                                # Create output directories if they don't exist
                                pdf_filtered_dir = default_dirs["input_pdf"]
                                md_dir = default_dirs["input_md"]
                                kg_dir = default_dirs["output_kg"]
                                final_dir = default_dirs["output_final"]
                                
                                os.makedirs(pdf_filtered_dir, exist_ok=True)
                                os.makedirs(md_dir, exist_ok=True)
                                os.makedirs(kg_dir, exist_ok=True)
                                os.makedirs(final_dir, exist_ok=True)

                                results = []
                                
                                logger.info("Step 1/5: Filtering small PDFs")
                                status.value = "Step 1/5: Filtering small PDFs..."
                                yield "\n".join(results) + f"\n{status.value}"
                                filtered_count = filter_small_pdfs(input_dir_val, pdf_filtered_dir, min_size_kb_val)
                                results.append(f"Filtered PDFs: {filtered_count} files saved to {pdf_filtered_dir}")
                                
                                logger.info("Step 2/5: Converting PDFs to Markdown")
                                status.value = "Step 2/5: Converting PDFs to Markdown..."
                                yield "\n".join(results) + f"\n{status.value}"
                                md_count = convert_pdfs_to_md(pdf_filtered_dir, md_dir, ollama_url_val, vision_model_val)
                                results.append(f"Converted to Markdown: {md_count} files saved to {md_dir}")
                                
                                logger.info("Step 3/5: Translating non-English Markdown files")
                                status.value = "Step 3/5: Translating Markdown files..."
                                yield "\n".join(results) + f"\n{status.value}"
                                translated_count = translate_markdown_files(md_dir) # Assumes output is same as input dir
                                results.append(f"Translated Markdown files: {translated_count} files processed in {md_dir}")
                                
                                logger.info("Step 4/5: Creating Knowledge Graph")
                                status.value = "Step 4/5: Creating Knowledge Graph..."
                                yield "\n".join(results) + f"\n{status.value}"
                                kg_path = os.path.join(kg_dir, f"knowledge_graph.{kg_format_val}")
                                kg_result = markdown_to_knowledge_graph(md_dir, kg_path, ollama_url_val, kg_model_val, 
                                                                       chunk_size_val, chunk_overlap_val, kg_format_val)
                                results.append(f"Knowledge Graph created: {kg_result} saved to {kg_path}")
                                
                                logger.info("Step 5/5: Post-processing Knowledge Graph")
                                status.value = "Step 5/5: Post-processing Knowledge Graph..."
                                yield "\n".join(results) + f"\n{status.value}"
                                pp_result, combined_nodes_info = postprocess_knowledge_graph(kg_path, final_dir, config=config)
                                results.append(f"Post-processed Knowledge Graph: saved to {final_dir}")
                                if combined_nodes_info:
                                    results.append("\n" + combined_nodes_info)
                                
                                logger.info("Pipeline completed")
                                status.value = "Pipeline completed."
                                yield "\n".join(results)
                            
                            run_button.click(
                                fn=run_pipeline,
                                inputs=[input_dir, output_dir, min_size_kb, ollama_url, vision_model, 
                                       kg_model, chunk_size, chunk_overlap, kg_format_type],
                                outputs=[output_text]
                            )
                sections.append(section_pipeline_ui)

                # Section: Filter PDFs
                with gr.Column(visible=False) as section_s1_ui:
                    gr.Markdown("### 1. Filter PDFs")
                    with gr.Row():
                        with gr.Column():
                            s1_input_dir = gr.Textbox(label="Input PDF Directory", value=default_dirs["input_all"], placeholder="Path to directory containing PDF files")
                            s1_output_dir = gr.Textbox(label="Output Directory", value=default_dirs["input_pdf"], placeholder="Path to save filtered PDFs")
                            s1_min_size_kb = gr.Slider(label="Minimum PDF Size (KB)", minimum=0, maximum=1000, value=config["parameters"]["min_size_kb"], step=10)
                        with gr.Column():
                            s1_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                            s1_run_button = gr.Button("Run Filtering", variant="primary")
                            s1_output_text = gr.Textbox(label="Output", interactive=False)
                            
                            def run_step1(input_dir_val, output_dir_val, min_size_kb_val):
                                if not input_dir_val or not output_dir_val:
                                    return "Please provide both input and output directories."
                                os.makedirs(output_dir_val, exist_ok=True)
                                s1_status.value = "Filtering..."
                                filtered_count = filter_small_pdfs(input_dir_val, output_dir_val, min_size_kb_val)
                                s1_status.value = "Done."
                                return f"Filtered PDFs: {filtered_count} files saved to {output_dir_val}"
                            
                            s1_run_button.click(
                                fn=run_step1,
                                inputs=[s1_input_dir, s1_output_dir, s1_min_size_kb],
                                outputs=[s1_output_text]
                            )
                sections.append(section_s1_ui)

                # Section: PDF to Markdown
                with gr.Column(visible=False) as section_s2_ui:
                    gr.Markdown("### 2. PDF to Markdown")
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
                            
                            def run_step2(input_dir_val, output_dir_val, ollama_url_val, ollama_model_val):
                                if not input_dir_val or not output_dir_val:
                                    return "Please provide both input and output directories."
                                os.makedirs(output_dir_val, exist_ok=True)
                                s2_status.value = "Converting..."
                                md_count = convert_pdfs_to_md(input_dir_val, output_dir_val, ollama_url_val, ollama_model_val)
                                s2_status.value = "Done."
                                return f"Converted to Markdown: {md_count} files saved to {output_dir_val}"
                            
                            s2_run_button.click(
                                fn=run_step2,
                                inputs=[s2_input_dir, s2_output_dir, s2_ollama_url, s2_ollama_model],
                                outputs=[s2_output_text]
                            )
                sections.append(section_s2_ui)

                # Section: Translate Markdown
                with gr.Column(visible=False) as section_s25_ui:
                    gr.Markdown("### 3. Translate Markdown")
                    with gr.Row():
                        with gr.Column():
                            s25_input_dir = gr.Textbox(label="Input Markdown Directory", value=default_dirs["input_md"], placeholder="Path to directory containing Markdown files")
                            s25_output_dir = gr.Textbox(label="Output Directory (Optional)", value=default_dirs["input_md"], placeholder="Path to save translated MD files (if different from input)")
                        with gr.Column():
                            s25_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                            s25_run_button = gr.Button("Translate Markdown", variant="primary")
                            s25_output_text = gr.Textbox(label="Output", interactive=False)
                            
                            def run_step25(input_dir_val, output_dir_val):
                                if not input_dir_val:
                                    return "Please provide input directory."
                                # If output_dir is not specified or same as input, files are processed in place or to input_dir
                                effective_output_dir = output_dir_val if output_dir_val and output_dir_val != input_dir_val else input_dir_val
                                if output_dir_val and output_dir_val != input_dir_val:
                                     os.makedirs(output_dir_val, exist_ok=True)
                                
                                s25_status.value = "Translating..."
                                translated_count = translate_markdown_files(input_dir_val, effective_output_dir)
                                s25_status.value = "Done."
                                return f"Translated Markdown files: {translated_count} files processed. Output to: {effective_output_dir}"
                            
                            s25_run_button.click(
                                fn=run_step25,
                                inputs=[s25_input_dir, s25_output_dir],
                                outputs=[s25_output_text]
                            )
                sections.append(section_s25_ui)
                
                # Section: Markdown to KG
                with gr.Column(visible=False) as section_s3_ui:
                    gr.Markdown("### 4. Markdown to KG")
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
                            
                            def run_step3(input_dir_val, output_file_val, ollama_url_val, ollama_model_val, chunk_size_val, chunk_overlap_val, format_type_val):
                                if not input_dir_val or not output_file_val:
                                    return "Please provide both input directory and output file."
                                os.makedirs(os.path.dirname(output_file_val), exist_ok=True)
                                # Update output file extension based on format type
                                output_file_val = Path(output_file_val).with_suffix(f".{format_type_val}")

                                s3_status.value = "Creating KG..."
                                kg_result = markdown_to_knowledge_graph(
                                    input_dir_val, str(output_file_val), ollama_url_val, ollama_model_val, 
                                    chunk_size_val, chunk_overlap_val, format_type_val
                                )
                                s3_status.value = "Done."
                                return f"Knowledge Graph created: {kg_result} saved to {output_file_val}"
                            
                            s3_run_button.click(
                                fn=run_step3,
                                inputs=[s3_input_dir, s3_output_file, s3_ollama_url, s3_ollama_model, 
                                       s3_chunk_size, s3_chunk_overlap, s3_format_type],
                                outputs=[s3_output_text]
                            )
                sections.append(section_s3_ui)

                # Section: Post-process KG
                with gr.Column(visible=False) as section_s4_ui:
                    gr.Markdown("### 5. Post-process Knowledge Graph")
                    with gr.Row():
                        with gr.Column():
                            s4_input_file = gr.Textbox(label="Input Knowledge Graph (.json or .xml)", value=os.path.join(default_dirs["output_kg"], "knowledge_graph.json"), placeholder="Path to the knowledge graph file")
                            s4_output_path = gr.Textbox(label="Output Directory", value=default_dirs["output_final"], placeholder="Path to save the processed knowledge graph files")
                        with gr.Column():
                            s4_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                            s4_run_button = gr.Button("Post-process Knowledge Graph", variant="primary")
                            s4_output_text = gr.Textbox(label="Output", interactive=False)
                            s4_combined_nodes_info = gr.Textbox(label="Redundant Nodes Eliminated", interactive=False, lines=5, max_lines=10)
                            
                            def run_step4(input_file_val, output_path_val):
                                if not input_file_val or not output_path_val:
                                    return "Please provide both input file and output path.", ""
                                os.makedirs(output_path_val, exist_ok=True) # output_path_val is a directory
                                s4_status.value = "Post-processing..."
                                output_info, combined_nodes_info_val = postprocess_knowledge_graph(input_file_val, output_path_val, config=config)
                                s4_status.value = "Done."
                                return f"Post-processed Knowledge Graph: {output_info} saved to {output_path_val}", combined_nodes_info_val or "No redundant nodes were eliminated."
                            
                            s4_run_button.click(
                                fn=run_step4,
                                inputs=[s4_input_file, s4_output_path],
                                outputs=[s4_output_text, s4_combined_nodes_info]
                            )
                sections.append(section_s4_ui)

                # Section: Visualize KG
                with gr.Column(visible=False) as section_viz_ui:
                    gr.Markdown("### Visualize Knowledge Graph")
                    with gr.Row():
                        with gr.Column():
                            viz_kg_dir = gr.Textbox(label="Knowledge Graph Directory", value=default_dirs["output_final"], placeholder="Path to directory containing final_kg files")
                            viz_show_contextual = gr.Checkbox(label="Show Contextual Proximity Edges", value=True, info="Toggle to show/hide contextual proximity edges")
                            viz_run_button = gr.Button("Visualize Knowledge Graph", variant="primary")
                            viz_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                    
                    with gr.Row():
                        viz_output = gr.Image(label="Knowledge Graph Visualization", interactive=False) # type="filepath" is default for gr.Image
                        
                        def visualize_kg(kg_dir_val, show_contextual_proximity_val):
                            if not kg_dir_val or not os.path.exists(kg_dir_val):
                                viz_status.value = "Error: KG directory not found."
                                return None, "Error: KG directory not found."
                            
                            finalgraph_path = os.path.join(kg_dir_val, "finalgraph.csv")
                            metadata_path = os.path.join(kg_dir_val, "metadata.csv")
                            
                            if not os.path.exists(finalgraph_path):
                                viz_status.value = "Error: finalgraph.csv not found in the directory."
                                return None, "Error: finalgraph.csv not found."
                            
                            viz_status.value = "Generating visualization..."
                            # Create visualization
                            viz_path = os.path.join(kg_dir_val, "visualization.png")

                            try:
                                create_knowledge_graph_visualization(
                                    finalgraph_path, 
                                    metadata_path if os.path.exists(metadata_path) else None,
                                    viz_path,
                                    figsize=(16, 14), # Increased size
                                    show_contextual_proximity=show_contextual_proximity_val
                                )
                                viz_status.value = f"Visualization saved to {viz_path}"
                                return viz_path, "Visualization complete."                                
                            except Exception as e:
                                logger.error(f"Error during visualization: {e}")
                                viz_status.value = f"Error: {e}"
                                return None, f"Error: {e}"
                        
                        viz_run_button.click(
                            fn=visualize_kg,
                            inputs=[viz_kg_dir, viz_show_contextual],
                            outputs=[viz_output, viz_status] # viz_output will display the image from the filepath
                        )
                sections.append(section_viz_ui)

                # Section: GraphRAG Chat
                with gr.Column(visible=False) as section_rag_ui:
                    gr.Markdown("### GraphRAG Chat")
                    gr.Markdown("""
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
                                                       placeholder="gemma3:12b") # Note: Original has 12b, config has 1b. Using provided code's default.
                            rag_query = gr.Textbox(label="Your Question", placeholder="Ask a question about the knowledge graph...", lines=2)
                            rag_run_button = gr.Button("Ask Question", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            rag_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                            rag_process = gr.Textbox(label="Process Notes", interactive=False, lines=10, max_lines=15)
                            rag_answer = gr.Textbox(label="Final Answer", interactive=False, lines=5, max_lines=10)
                        
                        def run_graphrag(kg_dir_val, query_val, ollama_url_val, model_val):
                            if not query_val or not kg_dir_val:
                                rag_status.value = "Error: Missing query or KG directory."
                                return "Please provide both a question and a knowledge graph directory.", "No query or knowledge graph specified.", "Error."
                            
                            rag_status.value = "Processing query with GraphRAG..."
                            try:
                                final_answer_val, process_notes_val = process_graph_query(kg_dir_val, query_val, ollama_url_val, model_val)
                                rag_status.value = "GraphRAG processing complete."
                                return process_notes_val, final_answer_val, "Complete."
                            except Exception as e:
                                logger.error(f"Error during GraphRAG: {e}")
                                rag_status.value = f"Error: {e}"
                                return f"Error during processing: {e}", "", f"Error: {e}"
                        
                        rag_run_button.click(
                            fn=run_graphrag,
                            inputs=[rag_kg_dir, rag_query, rag_ollama_url, rag_ollama_model],
                            outputs=[rag_process, rag_answer, rag_status]
                        )
                sections.append(section_rag_ui)

        # Function to control visibility of sections
        def set_visible_section(selected_index):
            return [gr.update(visible=(i == selected_index)) for i, _ in enumerate(sections)]

        # Connect sidebar buttons to the visibility function
        # The outputs list must match the order in the `sections` list
        sidebar_buttons = [b_pipeline, b_s1, b_s2, b_s25, b_s3, b_s4, b_viz, b_rag]
        for i, button in enumerate(sidebar_buttons):
            button.click(lambda idx=i: set_visible_section(idx), outputs=sections)
            
    return app

if __name__ == "__main__":
    app = create_app()
    config = load_config() # Load config again for server settings if needed, or pass from create_app
    app.launch(share=config["ui"].get("share", False), pwa=config["ui"].get("pwa", False))