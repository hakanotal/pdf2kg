from ollama import Client

import logging
import json
import sys
sys.path.append("..")

# Initialize the Ollama client
client = Client(host='http://169.226.53.98:11434')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_json_from_response(response_text):
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


def generate_graph(input: str, metadata={}, model="mistral-openorca:latest"):
    """Generate knowledge graph edges from input text using Ollama."""

    SYS_PROMPT = (
        "You are a knowledge graph expert that extracts terms and their relations from a given context.\n"
        "Your task is to identify key concepts and their relationships in the provided text.\n"
        "Guidelines:\n"
        "1. Identify two important terms (nodes) in the text including: objects, entities, locations, organizations, "
        "persons, conditions, documents, services, concepts, and dates.\n"
        "2. Determine relationships between pairs of terms that are mentioned in proximity.\n"
        "3. Dont forget to provide node types as well, use only the given types for node_1_type and node_2_type.\n"
        "4. Describe each relationship clearly and concisely.\n\n"
        "IMPORTANT: Your response MUST be a valid JSON object without any additional text or explanation. "
        "Format your output exactly as follows:\n"
        "{\n"
        '  "edges": [\n'
        "    {\n"
        '      "node_1": "Concept 1",\n'
        '      "node_1_type": "object|entity|location|organization|person|condition|document|concept|date|other",\n'
        '      "node_2": "Concept 2",\n'
        '      "node_2_type": "object|entity|location|organization|person|condition|document|concept|date|other",\n'
        '      "edge": "Relationship between the two concepts"\n'
        "    },\n"
        "    {...}\n"
        "  ]\n"
        "}\n"
    )

    USER_PROMPT = f"Context: ```{input}```"

    try:
        response = client.chat(
            model=model,
            stream=False,
            options={
                "temperature": 0,          
            },
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            format="json"
        )
        
        # Get the content from the last message
        response_content = response['message']['content']
        
        # Extract JSON from the response
        json_result = extract_json_from_response(response_content)
        
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
        logger.error(f"Error in graphPrompt: {e}")
        return None
    
