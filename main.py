#!/usr/bin/env python3

import os
import sys
import argparse
import socket
import yaml
from app.app import create_app, load_config

def create_default_directories(config):
    """Create default input and output directories if they don't exist."""
    print("Creating default directories:")
    
    # Create input directories
    input_dirs = [
        config["directories"]["input"]["all"],
        config["directories"]["input"]["pdf"],
        config["directories"]["input"]["md"]
    ]
    for dir_path in input_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"- {os.path.abspath(dir_path)}")
    
    # Create output directories
    output_dirs = [
        config["directories"]["output"]["kg"],
        config["directories"]["output"]["final"]
    ]
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"- {os.path.abspath(dir_path)}")

def is_port_in_use(host, port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def find_available_port(start_port=7860):
    """Find an available port starting from start_port."""
    port = start_port
    while is_port_in_use('127.0.0.1', port):
        port += 1
    return port

def main():
    """Main entry point for the PDF2KG application."""
    
    # Load configuration
    config = load_config()
    
    parser = argparse.ArgumentParser(description='PDF2KG - Convert PDF documents to Knowledge Graphs')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the Gradio interface on')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the Gradio interface on')
    parser.add_argument('--share', action='store_true', help='Create a publicly shareable link')
    parser.add_argument('--auto-port', action='store_true', help='Automatically find an available port if the specified one is in use')
    
    args = parser.parse_args()
    
    # Override config with command line arguments if provided
    share = args.share if args.share else config["ui"]["share"]
    
    # Create default directories
    create_default_directories(config)
    
    # Check if port is in use and find an available one if needed
    port = args.port
    if args.auto_port and is_port_in_use(args.host, args.port):
        port = find_available_port(args.port)
        print(f"Port {args.port} is already in use. Using port {port} instead.")
    
    print("Starting PDF2KG application...")
    print(f"Server will run on {args.host}:{port}")
    
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=port,
        share=share,
        pwa=config["ui"]["pwa"]
    )

if __name__ == "__main__":
    main() 