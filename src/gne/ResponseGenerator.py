"""
LlamaCppTextGenerator.py

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

# Get the functions at module level to avoid repeated calls
get_logger, _, _, _ = get_custom_logging()

class ResponseGenerator:
    """
    Response generator using llama.cpp for efficient inference.
    
    This class is specifically designed for generating responses to prompts
    using the prompt_template configuration from RGConfig.yaml.
    """
    
    _MODEL_CACHE = {}
    
    def __init__(self, model_key="response_generator", config_path="config/RGConfig.yaml", log_file: Optional[str] = None):
        self.log_file = log_file
        self.logger = get_logger("ResponseGenerator", self.log_file)
        self.logger.debug(f"Logger correctly initialized with log_file: {self.log_file}")

        # Load model config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if model_key not in config:
            raise ValueError(f"Model '{model_key}' not found in configuration.")
        self.model_cfg = config[model_key]

        # Response generator only needs its own config

        # Memory management settings
        self.enable_memory_cleanup = self.model_cfg.get("enable_memory_cleanup", True)
        self.max_memory_usage_gb = self.model_cfg.get("max_memory_usage_gb", 12.0)

        # Performance tracking attributes
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        
        # Load model
        model_path = self.model_cfg["name"]
        if model_path not in self._MODEL_CACHE:
            self.logger.info(f"Loading llama.cpp model: {model_path}")
            self._load_model(model_path)
        else:
            self.logger.info(f"Using cached llama.cpp model: {model_path}")

        self.model = self._MODEL_CACHE[model_path]
        self.generation_args = self.model_cfg.get("generation_args", {})

        # Prompt template support (for text-generation task)
        tmpl = self.model_cfg.get("prompt_template", {})
        self.prompt_format = tmpl.get("format", "{{prompt}}")
        self.user_prefix = tmpl.get("user_prefix", "")
        self.assistant_prefix = tmpl.get("assistant_prefix", "")
        
        # Response generator only uses prompt_template for response generation

        # Memory monitoring
        self.last_memory_check = time.time()
        self.memory_check_interval = 60  # Check memory every 60 seconds

    def _load_model(self, model_path: str):
        """Load model using llama.cpp with optimizations."""
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                # Try to find GGUF file
                gguf_path = f"{model_path}.gguf"
                if os.path.exists(gguf_path):
                    model_path = gguf_path
                else:
                    # For now, create a mock model for testing
                    self.logger.warning(f"Model file not found: {model_path}")
                    self.logger.info("Creating mock model for testing purposes")
                    mock_model = self._create_mock_model()
                    self._MODEL_CACHE[model_path] = mock_model
                    return
            
            # Configure llama.cpp parameters
            llama_params = {
                "model_path": model_path,
                "n_ctx": 2048,  # Context window
                "n_threads": None,  # Auto-detect
                "n_gpu_layers": 0,  # Use CPU for now, can be optimized later
                "verbose": False,
                "use_mmap": True,  # Memory mapping for efficiency
                "use_mlock": False,  # Don't lock memory
                "low_vram": False,  # Not needed for CPU
            }
            
            # Load model
            self.logger.info("Loading model with llama.cpp...")
            model = Llama(**llama_params)
            
            self._MODEL_CACHE[model_path] = model
            self.logger.info(f"Model loaded successfully: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Create mock model as fallback
            self.logger.info("Creating mock model as fallback")
            mock_model = self._create_mock_model()
            self._MODEL_CACHE[model_path] = mock_model

    def _create_mock_model(self):
        """Create a mock model for testing when real model is not available."""
        class MockModel:
            def __call__(self, prompt, **kwargs):
                # Return a mock response
                return {
                    'choices': [{
                        'text': f"Mock response to: {prompt[:50]}... [This is a test response from llama.cpp integration]"
                    }]
                }
        
        return MockModel()

    def _check_memory_usage(self):
        """Check memory usage and perform cleanup if needed."""
        current_time = time.time()
        if current_time - self.last_memory_check > self.memory_check_interval:
            try:
                memory_percent = psutil.virtual_memory().percent
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                
                self.logger.debug(f"Memory usage: {memory_percent:.1f}%, Available: {available_memory_gb:.1f}GB")
                
                # Perform cleanup if memory usage is high
                if memory_percent > 85:
                    self.logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
                    self._perform_memory_cleanup()
                
                self.last_memory_check = current_time
            except ImportError:
                self.logger.debug("psutil not available for memory monitoring")
            except Exception as e:
                self.logger.warning(f"Memory check failed: {e}")
    
    def _perform_memory_cleanup(self):
        """Perform memory cleanup."""
        try:
            import gc
            # Force garbage collection
            gc.collect()
            self.logger.info("Memory cleanup performed")
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

    def format_prompt(self, raw_prompt: str) -> str:
        """Format prompt using the configured template."""
        return (
            self.prompt_format
            .replace("{{user_prefix}}", self.user_prefix)
            .replace("{{assistant_prefix}}", self.assistant_prefix)
            .replace("{{prompt}}", raw_prompt)
        )

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response to a prompt using the prompt_template configuration."""
        start_time = time.time()
        
        try:
            # Use the prompt_template for response generation
            formatted_prompt = self.format_prompt(prompt)
            # Use generation args from RGConfig.yaml
            generation_kwargs = self.generation_args.copy()
            
            # Generate response using llama.cpp
            self.logger.debug(f"Generating response for prompt: {formatted_prompt[:100]}...")
            
            response = self.model(
                formatted_prompt,
                max_tokens=generation_kwargs.get("max_new_tokens", 2048),
                temperature=generation_kwargs.get("temperature", 0.7),
                top_p=generation_kwargs.get("top_p", 0.9),
                top_k=generation_kwargs.get("top_k", 40),
                repeat_penalty=generation_kwargs.get("repetition_penalty", 1.1),
                stop=["</s>", "<|endoftext|>"],  # Stop tokens
                echo=False,  # Don't echo the input
            )
            
            # Extract text from response
            if isinstance(response, dict) and 'choices' in response:
                generated_text = response['choices'][0]['text']
            elif isinstance(response, str):
                generated_text = response
            else:
                self.logger.warning(f"Unexpected response format: {type(response)}")
                generated_text = str(response)
            
            # Clean up the response
            generated_text = generated_text.strip()
            
            # Update performance metrics
            self.generation_count += 1
            self.total_tokens_generated += len(generated_text.split())
            
            self.logger.debug(f"Generated response: {generated_text[:100]}...")
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}", exc_info=True)
            return ""
        finally:
            end_time = time.time()
            generation_time = end_time - start_time
            # Store timing information
            if not hasattr(self, '_last_generation_time'):
                self._last_generation_time = {}
            self._last_generation_time['start_time'] = start_time
            self._last_generation_time['end_time'] = end_time
            self._last_generation_time['duration'] = generation_time


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
                'max_memory_limit_gb': self.max_memory_usage_gb,
                'memory_usage_percent': (process_memory.rss / (1024**3)) / self.max_memory_usage_gb * 100,
                'memory_cleanup_enabled': self.enable_memory_cleanup
            }
        except Exception as e:
            self.logger.warning(f"Failed to get memory stats: {e}")
            return {}

    def process_population(self, pop_path: str = "outputs/Population.json") -> None:
        """Process entire population for text generation one genome at a time."""
        try:
            self.logger.info("Starting population processing for text generation with llama.cpp")
            
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
                    response = self.generate_response(genome['prompt'])
                    
                    if response:
                        genome['model_name'] = self.model_cfg.get("name", "")
                        genome['generated_output'] = response
                        genome['status'] = 'pending_evaluation'
                        
                        # Add timing information to genome
                        if hasattr(self, '_last_generation_time'):
                            genome['generation_timing'] = self._last_generation_time.copy()
                        
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
            
            # Log final summary
            self.logger.info("Population processing completed:")
            self.logger.info("  - Total genomes: %d", len(population))
            self.logger.info("  - Processed: %d", total_processed)
            self.logger.info("  - Errors: %d", total_errors)
            
        except Exception as e:
            self.logger.error("Population processing failed: %s", e, exc_info=True)
            raise

    def _save_single_genome(self, genome: Dict[str, Any], pop_path: str) -> None:
        """Save a single genome immediately by updating the existing population file."""
        try:
            from pathlib import Path
            
            # Load existing population
            pop_path_obj = Path(pop_path)
            if not pop_path_obj.exists():
                self.logger.warning(f"Population file {pop_path} does not exist for single genome save")
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
                self.logger.warning(f"Genome {genome_id} not found in population for update")
                return
            
            # Save updated population
            with open(pop_path_obj, 'w', encoding='utf-8') as f:
                json.dump(population, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Immediately saved genome {genome_id} to {pop_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save single genome {genome.get('id', 'unknown')}: {e}")
