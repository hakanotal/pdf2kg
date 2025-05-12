import logging
import os
import yaml

def get_logger(name):
    """
    Get a logger configured based on the settings in config.yaml
    
    Args:
        name: Name of the logger (typically __name__)
    
    Returns:
        Logger instance
    """
    # Get logger for the specified name
    logger = logging.getLogger(name)
    
    # Skip if logger is already configured
    if logger.handlers:
        return logger
        
    # Load log level from config file
    log_level = get_log_level_from_config()
    
    # Configure logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False
    return logger

def get_log_level_from_config():
    """Read log level from config.yaml file"""
    try:
        # Find the config.yaml in the project root
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get log level from config
        log_level_str = config.get('settings', {}).get('log_level', 'INFO')
        
        # Convert string to logging level
        log_level = getattr(logging, log_level_str)
        return log_level
    except Exception as e:
        # Fall back to INFO level if any issue occurs
        print(f"Error loading log level from config: {e}. Using default INFO level.")
        return logging.INFO 