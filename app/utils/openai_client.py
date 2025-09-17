from openai import OpenAI
import json
import re
import xml.etree.ElementTree as ET
import os
import yaml
import time
from app.utils.logger import get_logger

# Get logger
logger = get_logger(__name__)

class OpenAIClient:
    """A wrapper for the OpenAI client with custom functionality."""

    def __init__(self):
        """Initialize the OpenAI client.
        
        Note: The OpenAI API key is read from the OPENAI_API_KEY environment variable.
        """
        try:
            # The OpenAI client is initialized without arguments, 
            # it will automatically look for the OPENAI_API_KEY environment variable.
            self.client = OpenAI()
            logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client. Make sure the OPENAI_API_KEY environment variable is set. Error: {e}")
            raise
            
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        """Load prompts from the YAML file in the project root directory."""
        # This method remains unchanged as it handles local file operations.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        prompts_path = os.path.join(project_root, "prompts.yaml")
        
        try:
            with open(prompts_path, 'r') as f:
                prompts = yaml.safe_load(f)
            logger.info(f"Successfully loaded prompts from {prompts_path}")
            return prompts
        except Exception as e:
            logger.error(f"Error loading prompts from {prompts_path}: {e}")
            return {}
            
    def query_llm(self, prompt, context="", model="gpt-4o-mini", temperature=0, max_retries=3, retry_delay=2):
        """General purpose method to query the OpenAI LLM.
        
        Args:
            prompt: The main instruction or question
            context: Additional context or information to include
            model: Name of the OpenAI model to use
            temperature: Sampling temperature (0.0 to 2.0)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            The generated text response from the LLM
        """
        if context:
            user_content = f"{prompt}\n\n{context}"
        else:
            user_content = prompt
            
        for attempt in range(max_retries):
            try:
                # Call the OpenAI API using the updated client and method
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": user_content}
                    ]
                )
                
                # Get the content from the response object following the OpenAI structure
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All retries failed: {str(e)}")
                    return f"Error: Could not generate response. {str(e)}"
        
        return ""

    def extract_json_from_response(self, response_text):
        """Extract JSON from the response text. This method remains largely unchanged."""
        try:
            # First try to parse the entire response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to find a JSON block in the text
            try:
                # Look for text between the first { and the last }
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                
                # Try to find a list format JSON
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                        
                raise ValueError("No JSON object or array found in the response")
            except Exception as e:
                logger.error(f"Failed to extract JSON: {e}")
                logger.error(f"Response text: {response_text}")
                raise

    def extract_xml_from_response(self, response_text):
        """Extract XML from the response text. This method remains unchanged."""
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

    def generate_graph(self, input_text, metadata={}, model="gpt-4o-mini", format_type="json"):
        """Generate knowledge graph edges from input text using OpenAI.
        
        Args:
            input_text: Text to extract knowledge graph from
            metadata: Additional metadata to add to the result
            model: Name of the OpenAI model to use
            format_type: Output format type ("json" or "xml")
            
        Returns:
            List of edge dictionaries or None on error
        """
        # Choose system prompt based on format type
        if format_type.lower() == "xml":
            sys_prompt = self.prompts.get("XML_KG_PROMPT", "")
            user_prompt = f"Context Text:\n{input_text}"
        else:
            sys_prompt = self.prompts.get("JSON_KG_PROMPT", "")
            # Ensure the system prompt for JSON requests instructs the model to only output JSON.
            sys_prompt += "\n\nIMPORTANT: You must only respond with a valid JSON object, and nothing else."
            user_prompt = f"Context: ```{input_text}```"
            format_type = "json"  # default to json if not xml
        
        try:
            # Prepare the API call parameters
            params = {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }

            # Use OpenAI's JSON mode for reliable JSON output
            if format_type.lower() == "json":
                params["response_format"] = {"type": "json_object"}

            # Call the OpenAI API
            response = self.client.chat.completions.create(**params)
            
            # Get the content from the response
            response_content = response.choices[0].message.content
            
            # Process response based on format type
            if format_type.lower() == "xml":
                triplets = self.extract_xml_from_response(response_content)
                if triplets:
                    return [dict(item, **metadata) for item in triplets]
                else:
                    logger.error("Failed to extract valid triplets from XML response")
                    return None
            else: # JSON format
                # Using JSON mode makes parsing much more reliable
                json_result = self.extract_json_from_response(response_content)
                if isinstance(json_result, dict) and 'edges' in json_result:
                    edges = json_result['edges']
                    return [dict(item, **metadata) for item in edges]
                else:
                    logger.error(f"Unexpected JSON structure, 'edges' key not found: {json_result}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in generate_graph: {e}")
            return None