import os
import sys
import gradio as gr
import tempfile
import yaml
from pathlib import Path

# Add the root directory to the path so we can import from the utils directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components.filter_small_pdfs import filter_small_pdfs
from app.components.pdf_to_md import convert_pdfs_to_md
from app.components.md_to_kg import markdown_to_knowledge_graph
from app.components.postprocess_kg import postprocess_knowledge_graph
from app.utils.visualize import create_knowledge_graph_visualization

def load_config():
    """Load configuration from the config.yaml file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    
    # Default configuration in case file doesn't exist or can't be read
    default_config = {
        "directories": {
            "input": {"all": "input/all", "pdf": "input/pdf", "md": "input/md"},
            "output": {"kg": "output/kg", "final": "output/final"}
        },
        "ollama": {
            "url": "http://localhost:11434",
            "models": {"vision": "llama3.2-vision:11b", "kg": "gemma3:1b"}
        },
        "parameters": {
            "min_size_kb": 50,
            "chunk_size": 1500,
            "chunk_overlap": 200
        },
        "ui": {
            "share": False,
            "pwa": True
        }
    }
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config file: {e}, using default configuration")
        return default_config

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
                        
                    with gr.Column():
                        status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        run_button = gr.Button("Run Complete Pipeline", variant="primary")
                        output_text = gr.Textbox(label="Output", interactive=False)
                        
                        def run_pipeline(input_dir, output_dir, min_size_kb, ollama_url, vision_model, kg_model, chunk_size, chunk_overlap):
                            if not input_dir or not output_dir:
                                return "Please provide both input and output directories."
                            
                            # Create output directories if they don't exist
                            pdf_filtered_dir = default_dirs["input_pdf"]
                            md_dir = default_dirs["input_md"]
                            kg_dir = default_dirs["output_kg"]
                            final_dir = default_dirs["output_final"]
                            
                            results = []
                            
                            # Step 1: Filter small PDFs
                            print("Step 1/4: Filtering small PDFs")
                            filtered_count = filter_small_pdfs(input_dir, pdf_filtered_dir, min_size_kb)
                            results.append(f"Filtered PDFs: {filtered_count} files saved to {pdf_filtered_dir}")
                            
                            # Step 2: Convert PDFs to Markdown
                            print("Step 2/4: Converting PDFs to Markdown")
                            md_count = convert_pdfs_to_md(pdf_filtered_dir, md_dir, ollama_url, vision_model)
                            results.append(f"Converted to Markdown: {md_count} files saved to {md_dir}")
                            
                            # Step 3: Create Knowledge Graph
                            print("Step 3/4: Creating Knowledge Graph")
                            kg_path = os.path.join(kg_dir, "knowledge_graph.json")
                            kg_result = markdown_to_knowledge_graph(md_dir, kg_path, ollama_url, kg_model, chunk_size, chunk_overlap)
                            results.append(f"Knowledge Graph created: {kg_result} saved to {kg_path}")
                            
                            # Step 4: Post-process Knowledge Graph
                            print("Step 4/4: Post-processing Knowledge Graph")
                            pp_result = postprocess_knowledge_graph(kg_path, final_dir)
                            results.append(f"Post-processed Knowledge Graph: saved to {final_dir}")
                            
                            print("Pipeline completed")
                            return "\n".join(results)
                        
                        run_button.click(
                            fn=run_pipeline,
                            inputs=[input_dir, output_dir, min_size_kb, ollama_url, vision_model, kg_model, chunk_size, chunk_overlap],
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
            
            with gr.TabItem("3. Markdown to KG"):
                with gr.Row():
                    with gr.Column():
                        s3_input_dir = gr.Textbox(label="Input Markdown Directory", value=default_dirs["input_md"], placeholder="Path to directory containing Markdown files")
                        s3_output_file = gr.Textbox(label="Output File", value=os.path.join(default_dirs["output_kg"], "knowledge_graph.json"), placeholder="Path to save the knowledge graph")
                        s3_ollama_url = gr.Textbox(label="Ollama Server URL", value=config["ollama"]["url"], placeholder="http://localhost:11434")
                        s3_ollama_model = gr.Textbox(label="Ollama Model Name", value=config["ollama"]["models"]["kg"], placeholder="gemma3:1b")
                        s3_chunk_size = gr.Slider(label="Chunk Size", minimum=500, maximum=3000, value=config["parameters"]["chunk_size"], step=100)
                        s3_chunk_overlap = gr.Slider(label="Chunk Overlap", minimum=0, maximum=500, value=config["parameters"]["chunk_overlap"], step=50)
                    with gr.Column():
                        s3_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        s3_run_button = gr.Button("Create Knowledge Graph", variant="primary")
                        s3_output_text = gr.Textbox(label="Output", interactive=False)
                        
                        def run_step3(input_dir, output_file, ollama_url, ollama_model, chunk_size, chunk_overlap):
                            if not input_dir or not output_file:
                                return "Please provide both input directory and output file."
                            os.makedirs(os.path.dirname(output_file), exist_ok=True)
                            kg_result = markdown_to_knowledge_graph(input_dir, output_file, ollama_url, ollama_model, chunk_size, chunk_overlap)
                            return f"Knowledge Graph created: {kg_result} saved to {output_file}"
                        
                        s3_run_button.click(
                            fn=run_step3,
                            inputs=[s3_input_dir, s3_output_file, s3_ollama_url, s3_ollama_model, s3_chunk_size, s3_chunk_overlap],
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
                        
                        def run_step4(input_file, output_path):
                            if not input_file or not output_path:
                                return "Please provide both input file and output path."
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            pp_result = postprocess_knowledge_graph(input_file, output_path)
                            return f"Post-processed Knowledge Graph: saved to {output_path}"
                        
                        s4_run_button.click(
                            fn=run_step4,
                            inputs=[s4_input_file, s4_output_path],
                            outputs=[s4_output_text]
                        )
            
            # Visualization Tab
            with gr.TabItem("5. Visualization"):
                with gr.Row():
                    with gr.Column(scale=3):
                        viz_kg_dir = gr.Textbox(label="Knowledge Graph Directory", value=default_dirs["output_final"], placeholder="Path to directory containing final_kg files")
                    with gr.Column(scale=1):
                        viz_run_button = gr.Button("Visualize Knowledge Graph", variant="primary")
                
                with gr.Row():
                    viz_output = gr.HTML(label="Interactive Knowledge Graph", value="<div style='height:700px'>Knowledge graph visualization will appear here after clicking the button above.</div>")
                    
                    def visualize_kg(kg_dir):
                        if not kg_dir or not os.path.exists(kg_dir):
                            return "<div class='error'>Please provide a valid knowledge graph directory</div>"
                        
                        # Look for finalgraph.csv and metadata.csv files
                        finalgraph_path = os.path.join(kg_dir, "finalgraph.csv")
                        metadata_path = os.path.join(kg_dir, "metadata.csv")
                        
                        if not os.path.exists(finalgraph_path):
                            return "<div class='error'>finalgraph.csv not found in the specified directory</div>"
                        
                        # Create visualization
                        viz_path = os.path.join(kg_dir, "visualization.html")
                        
                        result = create_knowledge_graph_visualization(
                            finalgraph_path, 
                            metadata_path if os.path.exists(metadata_path) else None,
                            viz_path
                        )
                        
                        if result:
                            # Read the generated HTML file content
                            try:
                                with open(viz_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                # Return the HTML content directly
                                return html_content
                            except Exception as e:
                                return f"<div class='error'>Error reading visualization file: {e}</div>"
                        else:
                            return "<div class='error'>Failed to create visualization</div>"
                    
                    viz_run_button.click(
                        fn=visualize_kg,
                        inputs=[viz_kg_dir],
                        outputs=[viz_output]
                    )
    
    return app

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    app = create_app()
    app.launch(share=config["ui"]["share"], pwa=config["ui"]["pwa"]) 