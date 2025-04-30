from ollama import Client
import json
import logging

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
    
    def generate_graph(self, input_text, metadata={}, model="gemma3:1b"):
        """Generate knowledge graph edges from input text using Ollama.
        
        Args:
            input_text: Text to extract knowledge graph from
            metadata: Additional metadata to add to the result
            model: Name of the Ollama model to use
            
        Returns:
            List of edge dictionaries or None on error
        """
        # System prompt for knowledge graph generation
        sys_prompt = (
            "You are a knowledge graph expert that extracts terms and their relations from a given context.\n"
            "Your task is to identify key concepts and their relationships in the provided text.\n"
            "Guidelines:\n"
            "1. Identify two important terms (nodes) in the text including objects, entities, locations, organizations, "
            "persons, conditions, documents, services, concepts, and dates.\n"
            "2. Determine relationships between pairs of terms that are mentioned in proximity.\n"
            "3. Describe each relationship clearly and concisely.\n\n"
            "IMPORTANT: Your response MUST be a valid JSON object without any additional text or explanation. "
            "Format your output exactly as follows:\n"
            "{\n"
            '  "edges": [\n'
            "    {\n"
            '      "node_1": "Concept 1",\n'
            '      "edge": "Relationship between the two nodes"\n'
            '      "node_2": "Concept 2",\n'
            "    },\n"
            "    {...}\n"
            "  ]\n"
            "}\n"
        )
        
        # User prompt with context
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
                ],
                format="json"
            )
            
            # Get the content from the last message
            response_content = response['message']['content']
            
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