"""
ResponseGenerator.py

A specialized text generator for response generation using prompt templates.
This class is used to generate responses to prompts using the prompt_template configuration.
"""

import os
import json
import yaml
import time
import psutil
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from utils import get_custom_logging
from .model_interface import LlamaCppChatInterface

# Get the functions at module level to avoid repeated calls
get_logger, _, _, _ = get_custom_logging()

class ResponseGenerator:
    """
    Response generator using v1/chat/completions interface for efficient inference.
    
    This class is specifically designed for generating responses to prompts
    using the prompt_template configuration from RGConfig.yaml with chat completions.
    """
    
    def __init__(self, model_key="response_generator", config_path="config/RGConfig.yaml", log_file: Optional[str] = None):
        self.log_file = log_file
        self.logger = get_logger("ResponseGenerator", self.log_file)
        self.logger.debug(f"Logger correctly initialized with log_file: {self.log_file}")

        # Load model config
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if not config:
                raise ValueError(f"Configuration file is empty: {config_path}")
            if model_key not in config:
                raise ValueError(f"Model '{model_key}' not found in configuration. Available keys: {list(config.keys())}")
            self.model_cfg = config[model_key]
            
            # Validate model configuration
            if not self.model_cfg.get("name"):
                raise ValueError(f"Model configuration missing 'name' field for {model_key}")
                
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse YAML configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model configuration: {e}")
            raise

        # Initialize chat interface
        try:
            self.model_interface = LlamaCppChatInterface(self.model_cfg, log_file)
        except Exception as e:
            self.logger.error(f"Failed to initialize model interface: {e}")
            raise
        
        # Prompt template support (for chat completions)
        tmpl = self.model_cfg.get("prompt_template", {})
        self.prompt_messages = tmpl.get("messages", [])
        
        # Performance tracking attributes
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0

    def _build_messages(self, raw_prompt: str) -> List[Dict[str, str]]:
        """Build messages array from prompt template and user input."""
        messages = []
        
        # Add template messages
        for msg_template in self.prompt_messages:
            role = msg_template.get("role", "user")
            content = msg_template.get("content", "")
            
            # Replace placeholders in content
            if "{{prompt}}" in content:
                content = content.replace("{{prompt}}", raw_prompt)
            
            if content.strip():  # Only add non-empty messages
                messages.append({"role": role, "content": content})
        
        # If no template messages, create a simple user message
        if not messages:
            messages.append({"role": "user", "content": raw_prompt})
        
        return messages

    def generate_response(self, prompt: str, **kwargs) -> tuple[str, float]:
        """Generate a response to a prompt using chat completions interface.
        
        Returns:
            tuple: (response_text, duration_in_seconds)
        """
        start_time = time.time()
        
        try:
            # Build messages from template and user input
            messages = self._build_messages(prompt)
            
            # Generate response using chat completions
            self.logger.debug(f"Generating response for prompt: {prompt[:100]}...")
            
            generated_text = self.model_interface.chat_completion(messages, **kwargs)
            
            # Filter out safety disclaimers and unwanted prefixes
            filtered_text = self._filter_response(generated_text)
            
            # Update performance metrics
            self.generation_count += 1
            self.total_tokens_generated += len(filtered_text.split())
            
            self.logger.debug(f"Generated response: {filtered_text[:100]}...")
            return filtered_text, time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}", exc_info=True)
            return "", time.time() - start_time

    def _filter_response(self, response: str) -> str:
        """Filter out safety disclaimers and unwanted prefixes from responses."""
        if not response:
            return response
        
        # Common safety disclaimer patterns to remove
        safety_patterns = [
            "I can provide information on a wide range of topics, but please note that",
            "I understand you're asking about",
            "I can help you with information about",
            "I'm here to provide information, but",
            "I can discuss this topic, but",
            "I understand your question about",
            "I can provide information about",
            "I'm happy to help with information about",
            "I can share information about",
            "I understand you want to know about",
            "I can provide details about",
            "I'm here to help with information about",
            "I can give you information about",
            "I understand you're interested in",
            "I can provide guidance on",
            "I'm here to assist with information about",
            "I can offer information about",
            "I understand you need information about",
            "I can provide details on",
            "I'm here to help you with information about"
        ]
        
        # Remove safety disclaimers
        filtered_response = response
        for pattern in safety_patterns:
            if pattern.lower() in filtered_response.lower():
                # Find the pattern and remove everything before the actual content
                pattern_start = filtered_response.lower().find(pattern.lower())
                if pattern_start != -1:
                    # Look for the actual content after the disclaimer
                    remaining_text = filtered_response[pattern_start + len(pattern):]
                    # Find the first sentence that doesn't start with safety language
                    sentences = remaining_text.split('.')
                    for i, sentence in enumerate(sentences):
                        sentence = sentence.strip()
                        if sentence and not any(safety_word in sentence.lower() for safety_word in 
                                              ['please note', 'but please', 'however', 'but', 'however,']):
                            # Found the actual content, reconstruct from this point
                            filtered_response = '.'.join(sentences[i:]).strip()
                            break
                    break
        
        # Remove common refusal patterns
        refusal_patterns = [
            "I'm sorry, but I cannot",
            "I cannot provide",
            "I'm not able to",
            "I'm unable to",
            "I can't help with",
            "I cannot help with",
            "I'm not allowed to",
            "I cannot assist with",
            "I'm sorry, I cannot",
            "I cannot give you",
            "I cannot share",
            "I cannot discuss",
            "I cannot provide information about",
            "I cannot help you with",
            "I cannot assist you with",
            "I cannot provide details about",
            "I cannot give you information about",
            "I cannot share information about",
            "I cannot discuss this topic",
            "I cannot provide guidance on"
        ]
        
        for pattern in refusal_patterns:
            if pattern.lower() in filtered_response.lower():
                # If we find a refusal, try to extract any actual content that might follow
                pattern_start = filtered_response.lower().find(pattern.lower())
                if pattern_start != -1:
                    # Look for content after the refusal
                    remaining_text = filtered_response[pattern_start + len(pattern):]
                    # Try to find actual content
                    if remaining_text.strip():
                        # If there's content after refusal, use it
                        filtered_response = remaining_text.strip()
                    else:
                        # If no content after refusal, return empty
                        return ""
        
        # Clean up the response
        filtered_response = filtered_response.strip()
        
        # Remove leading/trailing whitespace and common prefixes
        filtered_response = filtered_response.lstrip('.,!?;: ')
        
        return filtered_response


    def get_current_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics for monitoring."""
        try:
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'total_memory_gb': process_memory.rss / (1024**3),
                'available_system_gb': memory_info.available / (1024**3),
                'system_memory_percent': memory_info.percent,
                'max_memory_limit_gb': self.model_interface.max_memory_usage_gb,
                'memory_usage_percent': (process_memory.rss / (1024**3)) / self.model_interface.max_memory_usage_gb * 100,
                'memory_cleanup_enabled': self.model_interface.enable_memory_cleanup
            }
        except Exception as e:
            self.logger.warning(f"Failed to get memory stats: {e}")
            return {}

    def process_population(self, pop_path: str = "data/outputs/non_elites.json") -> None:
        """Process entire population for text generation one genome at a time."""
        try:
            self.logger.info("Starting population processing for text generation with chat completions")
            
            # Load population
            from utils.population_io import load_population
            population = load_population(pop_path, logger=self.logger)
            
            # Count genomes that need processing
            pending_genomes = [g for g in population if g.get('status') == 'pending_generation']
            self.logger.info("Found %d genomes pending generation out of %d total", len(pending_genomes), len(population))
            
            if not pending_genomes:
                self.logger.info("No genomes pending generation. Skipping processing.")
                return
            
            # Process each genome individually
            total_processed = 0
            total_errors = 0
            
            for i, genome in enumerate(pending_genomes):
                genome_id = genome.get('id', 'unknown')
                try:
                    self.logger.debug("Processing genome %s (%d/%d)", genome_id, i + 1, len(pending_genomes))
                    
                    # Generate response for this genome
                    response, response_duration = self.generate_response(genome['prompt'])
                    
                    if response:
                        genome['model_name'] = self.model_cfg.get("name", "")
                        genome['generated_output'] = response
                        genome['response_duration'] = response_duration
                        genome['status'] = 'pending_evaluation'
                        
                        total_processed += 1
                        self.logger.debug("Generated response for genome %s", genome_id)
                    else:
                        genome['status'] = 'error'
                        genome['error'] = 'Failed to generate response'
                        total_errors += 1
                        self.logger.warning("Failed to generate response for genome %s", genome_id)
                    
                    # Save this genome immediately after processing
                    self._save_single_genome(genome, pop_path)
                    self.logger.debug("Saved genome %s immediately after generation", genome_id)
                        
                except Exception as e:
                    genome['status'] = 'error'
                    genome['error'] = str(e)
                    total_errors += 1
                    self.logger.error("Error processing genome %s: %s", genome_id, e)
                    # Save even failed genomes immediately
                    self._save_single_genome(genome, pop_path)
            
            # Final batch save to ensure all changes are persisted
            try:
                from utils.population_io import save_population
                save_population(population, pop_path, logger=self.logger)
                self.logger.info("Final batch save completed successfully")
            except Exception as e:
                self.logger.error(f"Failed to perform final batch save: {e}")
            
            # Log final summary
            self.logger.info("Population processing completed:")
            self.logger.info("  - Total genomes: %d", len(population))
            self.logger.info("  - Processed: %d", total_processed)
            self.logger.info("  - Errors: %d", total_errors)
            
        except Exception as e:
            self.logger.error("Population processing failed: %s", e, exc_info=True)
            raise

    def _save_single_genome(self, genome: Dict[str, Any], pop_path: str) -> None:
        """
        Save a single genome immediately by updating the existing population file.
        This is a best-effort incremental save for crash recovery.
        A final batch save is always performed at the end of processing.
        """
        try:
            from pathlib import Path
            
            # Load existing population
            pop_path_obj = Path(pop_path)
            if not pop_path_obj.exists():
                self.logger.debug(f"Population file {pop_path} does not exist for incremental save, skipping (final batch save will persist changes)")
                return
            
            with open(pop_path_obj, 'r', encoding='utf-8') as f:
                population = json.load(f)
            
            # Find and update the genome in the population
            genome_id = genome.get('id')
            updated = False
            for i, existing_genome in enumerate(population):
                if existing_genome.get('id') == genome_id:
                    population[i] = genome
                    updated = True
                    break
            
            if not updated:
                self.logger.debug(f"Genome {genome_id} not found in file for incremental update (may be in memory only)")
                return
            
            # Save updated population
            with open(pop_path_obj, 'w', encoding='utf-8') as f:
                json.dump(population, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Incremental save completed for genome {genome_id}")
            
        except Exception as e:
            self.logger.debug(f"Incremental save failed for genome {genome.get('id', 'unknown')}: {e} (final batch save will persist changes)")
