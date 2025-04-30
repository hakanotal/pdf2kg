from ollama import Client
import json
import logging
import re
import xml.etree.ElementTree as ET
import os
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaClient:
    """A wrapper for the Ollama client with custom functionality."""
    
    def __init__(self, host="http://localhost:11434"):
        """Initialize the Ollama client.
        
        Args:
            host: Ollama server URL
        """
        self.client = Client(host=host)
        self.prompts = self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts from the YAML file."""
        prompts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.yaml")
        try:
            with open(prompts_path, 'r') as f:
                prompts = yaml.safe_load(f)
            logger.info(f"Successfully loaded prompts from {prompts_path}")
            return prompts
        except Exception as e:
            logger.error(f"Error loading prompts from {prompts_path}: {e}")
            return {}
            
    
    def extract_json_from_response(self, response_text):
        """Extract JSON from the response text, handling potential text before or after the JSON."""
        try:
            # First try to parse the entire response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to find JSON block in the text
            try:
                # Look for text between the first { and the last }
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                
                # Try to find a list format JSON
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                    
                raise ValueError("No JSON object or array found in the response")
            except Exception as e:
                logger.error(f"Failed to extract JSON: {e}")
                logger.error(f"Response text: {response_text}")
                raise
    
    def extract_xml_from_response(self, response_text):
        """Extract XML from the response text, handling potential text before or after the XML."""
        try:
            # Try to find XML content between <triplets> tags
            pattern = r'<triplets>.*?</triplets>'
            match = re.search(pattern, response_text, re.DOTALL)
            
            if match:
                xml_str = match.group(0)
                root = ET.fromstring(xml_str)
                triplets = []
                
                for triplet in root.findall('.//triplet'):
                    subject = triplet.find('subject').text
                    predicate = triplet.find('predicate').text
                    object_text = triplet.find('object').text
                    
                    triplets.append({
                        "node_1": subject,
                        "edge": predicate,
                        "node_2": object_text
                    })
                
                return triplets
            else:
                logger.error("No XML triplets found in the response")
                logger.error(f"Response text: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract XML: {e}")
            logger.error(f"Response text: {response_text}")
            return None
    
    def generate_graph(self, input_text, metadata={}, model="gemma3:1b", format_type="json"):
        """Generate knowledge graph edges from input text using Ollama.
        
        Args:
            input_text: Text to extract knowledge graph from
            metadata: Additional metadata to add to the result
            model: Name of the Ollama model to use
            format_type: Output format type ("json" or "xml")
            
        Returns:
            List of edge dictionaries or None on error
        """
        # Choose system prompt based on format type
        if format_type.lower() == "xml":
            sys_prompt = self.prompts.get("XML_KG_PROMPT", "")
        else:
            sys_prompt = self.prompts.get("JSON_KG_PROMPT", "")
            format_type = "json"  # default to json if not xml
        
        # User prompt with context
        if format_type.lower() == "xml":
            user_prompt = f"Context Text:\n{input_text}"
        else:
            user_prompt = f"Context: ```{input_text}```"
        
        try:
            # Call the Ollama API
            response = self.client.chat(
                model=model,
                stream=False,
                options={
                    "temperature": 0,          
                },
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Get the content from the last message
            response_content = response['message']['content']
            
            # Process response based on format type
            if format_type.lower() == "xml":
                # Extract XML from the response
                triplets = self.extract_xml_from_response(response_content)
                
                if triplets:
                    # Add metadata to each item
                    result = [dict(item, **metadata) for item in triplets]
                    return result
                else:
                    logger.error("Failed to extract valid triplets from XML response")
                    return None
            else:
                # Extract JSON from the response
                json_result = self.extract_json_from_response(response_content)
                
                # Get the edges from the result
                if isinstance(json_result, dict) and 'edges' in json_result:
                    result = json_result['edges']
                    # Add metadata to each item
                    result = [dict(item, **metadata) for item in result]
                    return result
                else:
                    logger.error(f"Unexpected JSON structure, 'edges' key not found: {json_result}")
                    return None
                
        except Exception as e:
            logger.error(f"Error in generate_graph: {e}")
            return None 