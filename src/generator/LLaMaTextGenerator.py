import os
import json
import torch
import yaml
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.custom_logging import get_logger, PerformanceLogger
from typing import List, Dict, Any, Optional
import time
import asyncio
from openai import AsyncOpenAI
from utils.population_io import load_population, save_population

class LlaMaTextGenerator:
    _MODEL_CACHE = {}
    _DEVICE_CACHE = None
    
    def __init__(self, model_key="llama", config_path="config/modelConfig.yaml", log_file: Optional[str] = None):
        self.log_file = log_file
        self.logger = get_logger("LLaMaTextGenerator", self.log_file)
        self.logger.debug(f"Logger correctly initialized with log_file: {self.log_file}")

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if model_key not in config:
            raise ValueError(f"Model '{model_key}' not found in configuration.")
        self.model_cfg = config[model_key]

        model_name = self.model_cfg["name"]
        if model_name not in self._MODEL_CACHE:
            self.logger.info(f"Loading LLaMA model: {model_name}")
            self._load_model_optimized(model_name)
        else:
            self.logger.info(f"Using cached LLaMA model: {model_name}")

        self.tokenizer, self.model, self.device = self._MODEL_CACHE[model_name]
        self.generation_args = self.model_cfg.get("generation_args", {})

        # Prompt template support
        tmpl = self.model_cfg.get("prompt_template", {})
        self.prompt_format = tmpl.get("format", "{{prompt}}")
        self.user_prefix = tmpl.get("user_prefix", "")
        self.assistant_prefix = tmpl.get("assistant_prefix", "")

        # Optimization settings
        self.max_batch_size = self.model_cfg.get("max_batch_size", 4)
        self.logger.info(f"Model loaded on {self.device} with batch size {self.max_batch_size}")

        # Performance tracking
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0

    def _get_optimal_device(self):
        """Get the best available device for M3 Mac"""
        if self._DEVICE_CACHE is not None:
            return self._DEVICE_CACHE
            
        if torch.backends.mps.is_available():
            self._DEVICE_CACHE = "mps"
            self.logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
        elif torch.cuda.is_available():
            self._DEVICE_CACHE = "cuda"
            self.logger.info("Using CUDA")
        else:
            self._DEVICE_CACHE = "cpu"
            self.logger.info("Using CPU")
        return self._DEVICE_CACHE

    def _load_model_optimized(self, model_name: str):
        """Load model with M3 Mac optimizations"""
        device = self._get_optimal_device()
        
        # Load tokenizer with optimizations
        self.logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            legacy=False,
            use_fast=True,  # Use fast tokenizer for better performance
            padding_side=self.model_cfg.get("padding_side", "left")  # Configurable padding direction
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Configure model loading for M3 Mac
        model_kwargs = {
            "torch_dtype": torch.float16,  # Use half precision for memory efficiency
            "low_cpu_mem_usage": True,
            "device_map": None  # We'll manually move to device
        }
        
        # Add quantization for memory efficiency on M3
        if device == "mps":
            # MPS doesn't support all quantization yet, use float16
            model_kwargs["torch_dtype"] = torch.float16
            self.logger.info("Using float16 for MPS optimization")
        elif device == "cuda":
            # Use 4-bit quantization if available
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                self.logger.info("Using 4-bit quantization for memory efficiency")
            except Exception as e:
                self.logger.warning(f"Quantization not available: {e}")
        
        self.logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Move to device and optimize
        if device != "cpu":
            model = model.to(device)
        model.eval()
        
        # Enable optimizations
        if hasattr(torch.backends, 'mps') and device == "mps":
            # MPS specific optimizations
            torch.backends.mps.allow_tf32 = True
        
        # Compile model for better performance (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead")
                self.logger.info("Model compiled for faster inference")
        except Exception as e:
            self.logger.warning(f"Model compilation failed: {e}")
        
        self._MODEL_CACHE[model_name] = (tokenizer, model, device)

    def format_prompt(self, raw_prompt: str) -> str:
        return (
            self.prompt_format
            .replace("{{user_prefix}}", self.user_prefix)
            .replace("{{assistant_prefix}}", self.assistant_prefix)
            .replace("{{prompt}}", raw_prompt)
        )

    def generate_response_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts in a single batch"""
        if not prompts:
            return []
        
        formatted_prompts = [self.format_prompt(prompt) for prompt in prompts]
        
        # Tokenize with padding for batch processing
        inputs = self.tokenizer(
            formatted_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048  # Reasonable limit for prompts
        ).to(self.device)
        
        # Generate with optimized settings
        generation_kwargs = {
            **self.generation_args,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,  # Enable KV cache for efficiency
            "do_sample": self.generation_args.get("do_sample", False)
        }
        
        with torch.no_grad():
            # Autocast (mixed-precision) is only valid on CUDA or MPS.  Using it
            # on CPU raises a RuntimeError, so we enable it conditionally.
            if self.device in ("cuda", "mps"):
                with torch.autocast(device_type=self.device, enabled=True):
                    outputs = self.model.generate(**inputs, **generation_kwargs)
            else:
                outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            decoded = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            # Extract only the generated part
            formatted_prompt = formatted_prompts[i]
            if decoded.startswith(formatted_prompt):
                response = decoded[len(formatted_prompt):].strip()
            else:
                response = decoded
            
            # Clean up response
            if 'Adult 2:' in response:
                response = response.split('Adult 2:')[-1].strip()
            
            responses.append(response)
        
        return responses

    def generate_response(self, prompt: str) -> str:
        """Single prompt generation (backwards compatibility)"""
        return self.generate_response_batch([prompt])[0]

    def process_population(self, pop_path: str = "outputs/Population.json", batch_size: int = None) -> None:
        """Process entire population for text generation with batch saving for fault tolerance"""
        # Use config batch size if not provided, fallback to default
        if batch_size is None:
            batch_size = self.model_cfg.get("generation_batch_size", 10)
        
        with PerformanceLogger(self.logger, "Process Population", pop_path=pop_path, batch_size=batch_size):
            try:
                self.logger.info("Starting population processing for text generation with batch saving")
                self.logger.info("Using batch size: %d (from config: %s)", batch_size, 
                               self.model_cfg.get("generation_batch_size", "default"))
                
                # Load population
                population = self._load_population(pop_path)
                
                # Count genomes that need processing
                pending_genomes = [g for g in population if g.get('status') == 'pending_generation']
                self.logger.info("Found %d genomes pending generation out of %d total", 
                               len(pending_genomes), len(population))
                
                if not pending_genomes:
                    self.logger.info("No genomes pending generation. Skipping processing.")
                    return
                
                # Process genomes in batches
                total_processed = 0
                total_errors = 0
                batch_count = 0
                
                for i in range(0, len(pending_genomes), batch_size):
                    batch_count += 1
                    batch_end = min(i + batch_size, len(pending_genomes))
                    batch_genomes = pending_genomes[i:batch_end]
                    
                    self.logger.info("Processing batch %d: genomes %d-%d", 
                                   batch_count, i + 1, batch_end)
                    
                    # Process each genome in the batch
                    batch_processed = 0
                    batch_errors = 0
                    
                    for genome in batch_genomes:
                        if genome.get('status') == 'pending_generation':
                            genome_id = genome.get('id', 'unknown')
                            self.logger.debug("Processing genome %s in batch %d", genome_id, batch_count)
                            
                            processed_genome = self._process_genome(genome)
                            
                            if processed_genome.get('status') == 'pending_evaluation':
                                batch_processed += 1
                            elif processed_genome.get('status') == 'error':
                                batch_errors += 1
                    
                    # Save population after each batch
                    if batch_processed > 0 or batch_errors > 0:
                        self.logger.info("Saving population after batch %d: %d processed, %d errors", 
                                       batch_count, batch_processed, batch_errors)
                        self._save_population(population, pop_path)
                    
                    total_processed += batch_processed
                    total_errors += batch_errors
                    
                    # Log batch summary
                    self.logger.info("Batch %d completed: %d processed, %d errors", 
                                   batch_count, batch_processed, batch_errors)
                
                # Log final summary
                self.logger.info("Population processing completed:")
                self.logger.info("  - Total batches: %d", batch_count)
                self.logger.info("  - Total genomes: %d", len(population))
                self.logger.info("  - Successfully processed: %d", total_processed)
                self.logger.info("  - Errors: %d", total_errors)
                self.logger.info("  - Skipped: %d", len(population) - total_processed - total_errors)
                
                # Log performance metrics
                if self.generation_count > 0:
                    avg_tokens = self.total_tokens_generated / self.generation_count
                    avg_time = self.total_generation_time / self.generation_count
                    self.logger.info("Generation Performance:")
                    self.logger.info("  - Total generations: %d", self.generation_count)
                    self.logger.info("  - Total tokens: %d", self.total_tokens_generated)
                    self.logger.info("  - Average tokens per generation: %.1f", avg_tokens)
                    self.logger.info("  - Average time per generation: %.3f seconds", avg_time)
                
            except Exception as e:
                self.logger.error("Population processing failed: %s", e, exc_info=True)
                raise

    def convert_text_to_tokens(self, text: str) -> List[int]:
        """Convert input text to its token IDs using the model's tokenizer."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0].tolist()
        return input_ids

    def convert_population_texts_to_tokens(self, pop_path="outputs/Population.json"):
        """Convert prompts and generated texts to token IDs for genomes
        with status 'pending_evolution' or 'complete'."""
        self.logger.info("Converting prompt and generated text to token IDs...")

        try:
            with open(pop_path, "r") as f:
                population = json.load(f)

            updated = False
            for genome in population:
                if genome.get("status") in ["pending_evolution", "complete"]:
                    prompt = genome.get("prompt", "")
                    response = genome.get("generated_text", "")
                    if prompt and response:
                        genome["input_tokens"] = self.convert_text_to_tokens(prompt)
                        genome["output_tokens"] = self.convert_text_to_tokens(response)
                        updated = True

            if updated:
                with open(pop_path, "w") as f:
                    json.dump(population, f, indent=2)
                self.logger.info("Updated population with input/output tokens.")
            else:
                self.logger.info("No genomes required token conversion.")
        except Exception as e:
            self.logger.error(f"Failed to convert texts to tokens: {e}")

    def paraphrase_text(self, text: str, num_variants: int = 2) -> List[str]:
        """Generate at most `num_variants` paraphrased versions of the input text."""
        self.logger.info("Generating up to %d paraphrased variants for input text", num_variants)

        instruction = f"Paraphrase the following statement in different ways, keeping the meaning intact:\n{text}"
        self.logger.debug("Received Instruction - %s", instruction)
        
        # Use batch generation for efficiency
        instructions = [instruction] * (num_variants * 2)  # Generate extra for uniqueness
        paraphrases = set()

        try:
            generated = self.generate_response_batch(instructions)
            for response in generated:
                if response.lower() != text.lower() and response.strip():
                    paraphrases.add(response.strip())
                    # Stop if we have enough unique paraphrases
                    if len(paraphrases) >= num_variants:
                        break
        except Exception as e:
            self.logger.error("Batch paraphrasing failed: %s", e, exc_info=True)
            return [text]

        paraphrase_list = list(paraphrases)[:num_variants]  # Limit to requested number
        self.logger.info("Generated %d unique paraphrased variants", len(paraphrase_list))
        return paraphrase_list if paraphrase_list else [text]

    def _load_population(self, pop_path: str) -> List[Dict[str, Any]]:
        return load_population(pop_path, logger=self.logger)

    def _save_population(self, population: List[Dict[str, Any]], pop_path: str) -> None:
        save_population(population, pop_path, logger=self.logger)

    def _generate_text_simulation(self, prompt: str, genome_id: str) -> str:
        """Simulate text generation with detailed logging"""
        with PerformanceLogger(self.logger, "Text Generation", genome_id=genome_id, prompt_length=len(prompt)):
            try:
                # Simulate generation time
                generation_time = 0.1 + (len(prompt) * 0.001)  # Simulate realistic timing
                time.sleep(generation_time)
                
                # Simulate different response patterns based on prompt content
                if "violence" in prompt.lower():
                    response = f"Simulated response for genome {genome_id}: I cannot and will not provide information about violence or harmful activities."
                elif "harmful" in prompt.lower():
                    response = f"Simulated response for genome {genome_id}: I'm designed to help, not harm. Let me assist you with something constructive."
                else:
                    response = f"Simulated response for genome {genome_id}: Here's a helpful and safe response to your query."
                
                # Update performance metrics
                self.generation_count += 1
                self.total_tokens_generated += len(response.split())
                self.total_generation_time += generation_time
                
                self.logger.debug("Generated response for genome %s: %d tokens in %.3f seconds", 
                                genome_id, len(response.split()), generation_time)
                
                return response
                
            except Exception as e:
                self.logger.error("Text generation failed for genome %s: %s", genome_id, e, exc_info=True)
                return f"Error generating response for genome {genome_id}: {str(e)}"
    
    def _process_genome(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single genome with comprehensive logging"""
        genome_id = genome.get('id', 'unknown')
        
        with PerformanceLogger(self.logger, "Process Genome", genome_id=genome_id):
            try:
                # Check if genome needs generation
                if genome.get('status') != 'pending_generation':
                    self.logger.debug("Skipping genome %s - status: %s", genome_id, genome.get('status'))
                    return genome
                
                self.logger.info("Processing genome %s for text generation", genome_id)
                
                # Extract prompt
                prompt = genome.get('prompt', '')
                if not prompt:
                    self.logger.warning("Empty prompt for genome %s", genome_id)
                    genome['status'] = 'error'
                    genome['error'] = 'Empty prompt'
                    return genome
                
                self.logger.debug("Generating text for genome %s with prompt length: %d", genome_id, len(prompt))
                
                # Check if simulation mode is enabled
                use_simulation = self.model_cfg.get("use_simulation", False)
                
                if use_simulation:
                    # Use simulation for testing
                    self.logger.info("Using simulation mode for genome %s", genome_id)
                    generated_text = self._generate_text_simulation(prompt, genome_id)
                else:
                    # Use real model generation
                    self.logger.info("Using real model generation for genome %s", genome_id)
                    try:
                        generated_text = self.generate_response(prompt)
                        # Update performance metrics for real generation
                        self.generation_count += 1
                        self.total_tokens_generated += len(generated_text.split())
                        # Note: generation time is tracked in generate_response_batch
                    except Exception as e:
                        # Enhanced error logging with prompt text
                        log_failing_prompts = self.model_cfg.get("log_failing_prompts", True)
                        if log_failing_prompts:
                            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
                            self.logger.error("Generation failed for genome %s. Prompt preview: %s. Error: %s", 
                                            genome_id, prompt_preview, str(e), exc_info=True)
                        else:
                            self.logger.error("Generation failed for genome %s. Error: %s", 
                                            genome_id, str(e), exc_info=True)
                        genome['status'] = 'error'
                        genome['error'] = f"Generation failed: {str(e)}"
                        return genome
                
                # Update genome
                genome['generated_text'] = generated_text
                genome['status'] = 'pending_evaluation'
                genome['generation_timestamp'] = time.time()
                genome['model_provider'] = self.model_cfg.get("provider", "")
                genome['model_name'] = self.model_cfg.get("name", "")
                
                self.logger.info("Successfully generated text for genome %s: %d characters", 
                               genome_id, len(generated_text))
                
                return genome
                
            except Exception as e:
                # Enhanced error logging with prompt text
                log_failing_prompts = self.model_cfg.get("log_failing_prompts", True)
                if log_failing_prompts and 'prompt' in locals():
                    prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
                    self.logger.error("Failed to process genome %s. Prompt preview: %s. Error: %s", 
                                    genome_id, prompt_preview, str(e), exc_info=True)
                else:
                    self.logger.error("Failed to process genome %s. Error: %s", 
                                    genome_id, str(e), exc_info=True)
                genome['status'] = 'error'
                genome['error'] = str(e)
                return genome
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the generator"""
        stats = {
            'generation_count': self.generation_count,
            'total_tokens_generated': self.total_tokens_generated,
            'total_generation_time': self.total_generation_time,
            'model_name': self.model_cfg.get("name", "Unknown")
        }
        
        if self.generation_count > 0:
            stats['average_tokens_per_generation'] = self.total_tokens_generated / self.generation_count
            stats['average_time_per_generation'] = self.total_generation_time / self.generation_count
            stats['tokens_per_second'] = self.total_tokens_generated / self.total_generation_time
        
        self.logger.debug("Performance stats: %s", stats)
        return stats