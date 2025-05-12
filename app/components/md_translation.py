import os
import re
from pathlib import Path
from typing import List, Optional
from app.utils.logger import get_logger
from deep_translator import GoogleTranslator

# Get logger
logger = get_logger(__name__)

def translate_text(text: str, target_lang: str = 'en') -> str:
    """
    Translate text to the target language using Google Translate with auto language detection.
    
    Args:
        text (str): Text to translate
        target_lang (str): Target language code (default: 'en')
        
    Returns:
        str: Translated text
    """
    # Split text into chunks of max 5000 characters to respect Google Translate limits
    # But try to split on sentence boundaries for better translation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds the limit, add the current chunk to chunks and start a new one
        if len(current_chunk) + len(sentence) + 1 > 4800:  # Using 4800 to leave some margin
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # Translate each chunk
    translator = GoogleTranslator(source='auto', target=target_lang)
    translated_chunks = []
    
    for i, chunk in enumerate(chunks):
        try:
            logger.info(f"Translating chunk {i+1}/{len(chunks)}")
            translated_chunk = translator.translate(chunk)
            translated_chunks.append(translated_chunk)
        except Exception as e:
            logger.error(f"Translation error for chunk {i+1}: {e}")
            # If translation fails, keep the original text for this chunk
            translated_chunks.append(chunk)
    
    # Join the translated chunks back together
    return " ".join(translated_chunks)

def translate_markdown_files(input_dir: str, output_dir: Optional[str] = None, 
                            progress: Optional[callable] = None) -> int:
    """
    Check if markdown files are in English, and if not, translate them.
    
    Args:
        input_dir (str): Directory containing markdown files
        output_dir (str, optional): Output directory for translated files. If None, files are overwritten.
        progress (callable, optional): Progress callback function
        
    Returns:
        int: Number of translated files
    """
    # If output_dir is not provided, use input_dir (overwrite files)
    if output_dir is None:
        output_dir = input_dir
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all markdown files
    md_files = list(Path(input_dir).glob('**/*.md'))
    
    if not md_files:
        logger.warning(f"No markdown files found in {input_dir}")
        return 0
    
    logger.info(f"Found {len(md_files)} markdown files in {input_dir}")
    
    translated_count = 0
    
    # Process each file
    for i, md_file in enumerate(md_files):
        if progress:
            # Calculate progress from 0 to 1
            current_progress = i / len(md_files)
            progress(current_progress, desc=f"Processing file {i+1}/{len(md_files)}")
        
        file_path = str(md_file)
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name) if output_dir != input_dir else file_path
        
        logger.info(f"Processing {file_name}")
        
        try:
            # Read the markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Translate the content (GoogleTranslator will auto-detect the language)
            # If the text is already in English, Google will return it unchanged
            translated_content = translate_text(content)
            
            # Write translated content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            
            translated_count += 1
            logger.info(f"Processed {file_name} successfully")
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
    
    if progress:
        progress(1.0, desc=f"Translation complete: {translated_count} files processed")
    
    logger.info(f"Translation complete: {translated_count} files processed")
    
    return translated_count 