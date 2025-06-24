import yaml
import os
import json
from typing import Dict, Any, Optional
from utils.custom_logging import get_logger, PerformanceLogger
import logging

def load_config(config_path: str, log_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file with comprehensive logging"""
    logger = get_logger("config", log_file)
    
    with PerformanceLogger(logger, "Load Config", config_path=config_path):
        try:
            logger.info("Loading configuration from: %s", config_path)
            
            # Check if config file exists
            if not os.path.exists(config_path):
                logger.error("Configuration file not found: %s", config_path)
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Load YAML file
            with PerformanceLogger(logger, "Parse YAML Config"):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    logger.info("Successfully loaded configuration from %s", config_path)
                    logger.debug("Configuration keys: %s", list(config.keys()) if config else [])
                    
                except yaml.YAMLError as e:
                    logger.error("Failed to parse YAML configuration: %s", e, exc_info=True)
                    raise
                except Exception as e:
                    logger.error("Failed to read configuration file: %s", e, exc_info=True)
                    raise
            
            # Validate configuration
            with PerformanceLogger(logger, "Validate Config"):
                try:
                    validation_result = validate_config(config)
                    
                    if validation_result['is_valid']:
                        logger.info("Configuration validation successful")
                    else:
                        logger.warning("Configuration validation issues found:")
                        for issue in validation_result['issues']:
                            logger.warning("  - %s", issue)
                    
                    # Log configuration summary
                    logger.info("Configuration summary:")
                    for key, value in config.items():
                        if isinstance(value, dict):
                            logger.info("  %s: %d sub-keys", key, len(value))
                        elif isinstance(value, list):
                            logger.info("  %s: %d items", key, len(value))
                        else:
                            logger.info("  %s: %s", key, str(value)[:100] + "..." if len(str(value)) > 100 else str(value))
                    
                    return config
                    
                except Exception as e:
                    logger.error("Failed to validate configuration: %s", e, exc_info=True)
                    raise
            
        except Exception as e:
            logger.error("Configuration loading failed: %s", e, exc_info=True)
            raise

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration structure and values with detailed logging"""
    logger = get_logger("config_validation")
    
    with PerformanceLogger(logger, "Validate Configuration"):
        try:
            validation_result = {
                'is_valid': True,
                'issues': [],
                'warnings': []
            }
            
            # Check if config is not None
            if config is None:
                validation_result['is_valid'] = False
                validation_result['issues'].append("Configuration is None")
                return validation_result
            
            # Check required top-level keys
            required_keys = ['model', 'evolution', 'evaluation']
            for key in required_keys:
                if key not in config:
                    validation_result['is_valid'] = False
                    validation_result['issues'].append(f"Missing required key: {key}")
                elif not isinstance(config[key], dict):
                    validation_result['is_valid'] = False
                    validation_result['issues'].append(f"Key '{key}' must be a dictionary")
            
            # Validate model configuration
            if 'model' in config and isinstance(config['model'], dict):
                model_validation = validate_model_config(config['model'])
                validation_result['issues'].extend(model_validation['issues'])
                validation_result['warnings'].extend(model_validation['warnings'])
            
            # Validate evolution configuration
            if 'evolution' in config and isinstance(config['evolution'], dict):
                evolution_validation = validate_evolution_config(config['evolution'])
                validation_result['issues'].extend(evolution_validation['issues'])
                validation_result['warnings'].extend(evolution_validation['warnings'])
            
            # Validate evaluation configuration
            if 'evaluation' in config and isinstance(config['evaluation'], dict):
                evaluation_validation = validate_evaluation_config(config['evaluation'])
                validation_result['issues'].extend(evaluation_validation['issues'])
                validation_result['warnings'].extend(evaluation_validation['warnings'])
            
            # Update overall validity
            validation_result['is_valid'] = len(validation_result['issues']) == 0
            
            logger.debug("Configuration validation completed: %d issues, %d warnings", 
                        len(validation_result['issues']), len(validation_result['warnings']))
            
            return validation_result
            
        except Exception as e:
            logger.error("Configuration validation failed: %s", e, exc_info=True)
            return {
                'is_valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': []
            }

def validate_model_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate model configuration section"""
    logger = get_logger("model_config_validation")
    
    validation_result = {
        'issues': [],
        'warnings': []
    }
    
    try:
        # Check required model keys
        required_model_keys = ['name', 'type']
        for key in required_model_keys:
            if key not in model_config:
                validation_result['issues'].append(f"Model config missing required key: {key}")
        
        # Validate model name
        if 'name' in model_config:
            model_name = model_config['name']
            if not isinstance(model_name, str) or not model_name.strip():
                validation_result['issues'].append("Model name must be a non-empty string")
        
        # Validate model type
        if 'type' in model_config:
            model_type = model_config['type']
            valid_types = ['llama', 'gpt', 'claude', 'custom']
            if model_type not in valid_types:
                validation_result['warnings'].append(f"Model type '{model_type}' not in standard types: {valid_types}")
        
        # Validate parameters
        if 'parameters' in model_config:
            params = model_config['parameters']
            if not isinstance(params, dict):
                validation_result['issues'].append("Model parameters must be a dictionary")
            else:
                # Check for common parameter types
                for param_name, param_value in params.items():
                    if not isinstance(param_name, str):
                        validation_result['issues'].append(f"Parameter name must be string: {param_name}")
                    if not isinstance(param_value, (int, float, str, bool)):
                        validation_result['warnings'].append(f"Parameter '{param_name}' has non-standard type: {type(param_value)}")
        
        logger.debug("Model configuration validation: %d issues, %d warnings", 
                    len(validation_result['issues']), len(validation_result['warnings']))
        
    except Exception as e:
        logger.error("Model configuration validation failed: %s", e, exc_info=True)
        validation_result['issues'].append(f"Model validation error: {str(e)}")
    
    return validation_result

def validate_evolution_config(evolution_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate evolution configuration section"""
    logger = get_logger("evolution_config_validation")
    
    validation_result = {
        'issues': [],
        'warnings': []
    }
    
    try:
        # Check for common evolution parameters
        common_params = ['population_size', 'mutation_rate', 'crossover_rate', 'generations']
        
        for param in common_params:
            if param in evolution_config:
                value = evolution_config[param]
                
                if param in ['population_size', 'generations']:
                    if not isinstance(value, int) or value <= 0:
                        validation_result['issues'].append(f"{param} must be a positive integer, got: {value}")
                
                elif param in ['mutation_rate', 'crossover_rate']:
                    if not isinstance(value, (int, float)) or value < 0 or value > 1:
                        validation_result['issues'].append(f"{param} must be a number between 0 and 1, got: {value}")
        
        # Validate operators
        if 'operators' in evolution_config:
            operators = evolution_config['operators']
            if not isinstance(operators, dict):
                validation_result['issues'].append("Evolution operators must be a dictionary")
            else:
                valid_operator_types = ['mutation', 'crossover', 'selection']
                for op_type, op_config in operators.items():
                    if op_type not in valid_operator_types:
                        validation_result['warnings'].append(f"Unknown operator type: {op_type}")
                    if not isinstance(op_config, dict):
                        validation_result['issues'].append(f"Operator '{op_type}' configuration must be a dictionary")
        
        logger.debug("Evolution configuration validation: %d issues, %d warnings", 
                    len(validation_result['issues']), len(validation_result['warnings']))
        
    except Exception as e:
        logger.error("Evolution configuration validation failed: %s", e, exc_info=True)
        validation_result['issues'].append(f"Evolution validation error: {str(e)}")
    
    return validation_result

def validate_evaluation_config(evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate evaluation configuration section"""
    logger = get_logger("evaluation_config_validation")
    
    validation_result = {
        'issues': [],
        'warnings': []
    }
    
    try:
        # Check for evaluation metrics
        if 'metrics' in evaluation_config:
            metrics = evaluation_config['metrics']
            if not isinstance(metrics, list):
                validation_result['issues'].append("Evaluation metrics must be a list")
            else:
                for i, metric in enumerate(metrics):
                    if not isinstance(metric, str):
                        validation_result['issues'].append(f"Metric {i} must be a string, got: {type(metric)}")
        
        # Validate API configuration
        if 'api' in evaluation_config:
            api_config = evaluation_config['api']
            if not isinstance(api_config, dict):
                validation_result['issues'].append("API configuration must be a dictionary")
            else:
                required_api_keys = ['endpoint', 'timeout']
                for key in required_api_keys:
                    if key not in api_config:
                        validation_result['warnings'].append(f"API config missing recommended key: {key}")
        
        logger.debug("Evaluation configuration validation: %d issues, %d warnings", 
                    len(validation_result['issues']), len(validation_result['warnings']))
        
    except Exception as e:
        logger.error("Evaluation configuration validation failed: %s", e, exc_info=True)
        validation_result['issues'].append(f"Evaluation validation error: {str(e)}")
    
    return validation_result

def save_config(config: Dict[str, Any], config_path: str, log_file: Optional[str] = None) -> None:
    """Save configuration to YAML file with comprehensive logging"""
    logger = get_logger("config", log_file)
    
    with PerformanceLogger(logger, "Save Config", config_path=config_path):
        try:
            logger.info("Saving configuration to: %s", config_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Validate configuration before saving
            validation_result = validate_config(config)
            if not validation_result['is_valid']:
                logger.warning("Saving configuration with validation issues:")
                for issue in validation_result['issues']:
                    logger.warning("  - %s", issue)
            
            # Save configuration
            with PerformanceLogger(logger, "Write Config File"):
                try:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
                    
                    logger.info("Successfully saved configuration to %s", config_path)
                    
                    # Log file size
                    file_size = os.path.getsize(config_path)
                    logger.debug("Configuration file size: %d bytes", file_size)
                    
                except Exception as e:
                    logger.error("Failed to write configuration file: %s", e, exc_info=True)
                    raise
            
        except Exception as e:
            logger.error("Configuration saving failed: %s", e, exc_info=True)
            raise

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get configuration value using dot notation with comprehensive logging"""
    logger = get_logger("config_getter")
    
    try:
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                logger.debug("Configuration key '%s' not found, using default: %s", key_path, default)
                return default
        
        logger.debug("Retrieved configuration value for '%s': %s", key_path, value)
        return value
        
    except Exception as e:
        logger.error("Failed to get configuration value for '%s': %s", key_path, e, exc_info=True)
        return default

def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> bool:
    """Set configuration value using dot notation with comprehensive logging"""
    logger = get_logger("config_setter")
    
    try:
        keys = key_path.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        
        logger.debug("Set configuration value for '%s': %s", key_path, value)
        return True
        
    except Exception as e:
        logger.error("Failed to set configuration value for '%s': %s", key_path, e, exc_info=True)
        return False