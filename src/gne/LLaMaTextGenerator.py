"""
LLaMaTextGenerator.py
"""

import os
import json
import torch
import yaml
import gc
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import get_custom_logging
from utils.device_utils import get_optimal_device, get_device_info, move_to_device, get_generation_kwargs
from typing import List, Dict, Any, Optional
import time

# Get the functions at module level to avoid repeated calls
get_logger, _, _, _ = get_custom_logging()

class LlaMaTextGenerator:
    _MODEL_CACHE = {}
    _DEVICE_CACHE = None
    
    def __init__(self, model_key="llama", config_path="../config/modelConfig.yaml", log_file: Optional[str] = None):
        self.log_file = log_file
        self.logger = get_logger("LLaMaTextGenerator", self.log_file)
        self.logger.debug(f"Logger correctly initialized with log_file: {self.log_file}")

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if model_key not in config:
            raise ValueError(f"Model '{model_key}' not found in configuration.")
        self.model_cfg = config[model_key]

        # Memory management settings
        self.enable_memory_cleanup = self.model_cfg.get("enable_memory_cleanup", True)
        self.max_memory_usage_gb = self.model_cfg.get("max_memory_usage_gb", 12.0)  # Increased to 12GB

        # Performance tracking attributes
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        
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
        
        # Task-specific templates and generation args
        self.task_templates = self.model_cfg.get("task_templates", {})
        self.task_generation_args = self.model_cfg.get("task_generation_args", {})

        # Optimization settings
        self.logger.debug(f"Model loaded on {self.device}")

        # Memory monitoring
        self.last_memory_check = time.time()
        self.memory_check_interval = 60  # Check memory every 60 seconds

    def _check_memory_usage(self):
        """Check memory usage and perform cleanup if needed"""
        current_time = time.time()
        if current_time - self.last_memory_check > self.memory_check_interval:
            try:
                import psutil
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
        """Perform aggressive memory cleanup"""
        try:
            import gc
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Memory cleanup performed")
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
    
    def _get_optimal_device(self):
        """Get the best available device using centralized device manager"""
        if self._DEVICE_CACHE is not None:
            return self._DEVICE_CACHE
        
        self._DEVICE_CACHE = get_optimal_device()
        return self._DEVICE_CACHE

    def _load_model_optimized(self, model_name: str):
        """Load model with M3 Mac optimizations and timeout protection"""
        device = self._get_optimal_device()
        
        # Load tokenizer with optimizations
        self.logger.info("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                legacy=False,
                use_fast=True,  # Use fast tokenizer for better performance
                padding_side=self.model_cfg.get("padding_side", "left"),  # Configurable padding direction
                timeout=60  # Add timeout to prevent hanging
            )
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
        
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
        
        # Add timeout protection for model loading
        import signal
        import threading
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timed out after 300 seconds")
        
        # Set up timeout
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minute timeout
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            signal.alarm(0)  # Cancel timeout
        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            self.logger.error("Model loading timed out. Trying with CPU fallback...")
            # Fallback to CPU if MPS hangs
            if device == "mps":
                self.logger.info("Falling back to CPU for model loading")
                model_kwargs["torch_dtype"] = torch.float32  # Use float32 for CPU
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                device = "cpu"
                self._DEVICE_CACHE = "cpu"
            else:
                raise
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            self.logger.error(f"Failed to load model: {e}")
            raise
        finally:
            signal.signal(signal.SIGALRM, original_handler)
        
        # Move to device and optimize with centralized device management
        try:
            model = move_to_device(model, device)
            self.logger.info(f"Model successfully moved to {device}")
        except Exception as e:
            self.logger.error(f"Failed to move model to device: {e}")
            raise RuntimeError(f"Unable to load model on any device: {e}")
        
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

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB with enhanced monitoring"""
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        # Enhanced GPU memory monitoring
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_memory_max = torch.cuda.max_memory_allocated() / (1024**3)
        elif self.device == "mps":
            # MPS doesn't provide detailed memory info, use system memory
            gpu_memory = 0.0
            gpu_memory_reserved = 0.0
            gpu_memory_max = 0.0
        else:
            gpu_memory = 0.0
            gpu_memory_reserved = 0.0
            gpu_memory_max = 0.0
            
        return {
            "cpu_memory_gb": memory_info.rss / (1024**3),
            "gpu_memory_gb": gpu_memory,
            "gpu_memory_reserved_gb": gpu_memory_reserved,
            "gpu_memory_max_gb": gpu_memory_max,
            "total_memory_gb": memory_info.rss / (1024**3) + gpu_memory,
            "available_system_gb": virtual_memory.available / (1024**3),
            "system_memory_percent": virtual_memory.percent
        }

    def _check_memory_and_cleanup(self, force: bool = False) -> bool:
        """Enhanced memory usage check and cleanup with real-time monitoring"""
        current_time = time.time()
        
        # Only check periodically unless forced
        if not force and (current_time - self.last_memory_check) < self.memory_check_interval:
            return True
            
        self.last_memory_check = current_time
        memory_usage = self._get_memory_usage()
        
        self.logger.debug(f"Memory usage: {memory_usage}")
        
        # Enhanced memory threshold checking with 12GB limit
        memory_warning_threshold = self.max_memory_usage_gb * 0.75  # Warning at 75% (9GB)
        memory_critical_threshold = self.max_memory_usage_gb * 0.90  # Critical at 90% (10.8GB)
        
        # Check system memory availability
        if memory_usage["available_system_gb"] < 2.0:  # Less than 2GB available
            self.logger.warning(f"System memory critically low: {memory_usage['available_system_gb']:.2f}GB available")
            if self.enable_memory_cleanup:
                self._cleanup_memory(aggressive=True)
                return True
        
        # Check total memory usage
        if memory_usage["total_memory_gb"] > memory_critical_threshold:
            self.logger.error(f"CRITICAL: Memory usage ({memory_usage['total_memory_gb']:.2f}GB) exceeds critical threshold ({memory_critical_threshold:.2f}GB)")
            if self.enable_memory_cleanup:
                self._cleanup_memory(aggressive=True)
                return True
            else:
                return False
        elif memory_usage["total_memory_gb"] > memory_warning_threshold:
            self.logger.warning(f"Memory usage ({memory_usage['total_memory_gb']:.2f}GB) exceeds warning threshold ({memory_warning_threshold:.2f}GB)")
            if self.enable_memory_cleanup:
                self._cleanup_memory()
                return True
                
        return True

    def _cleanup_memory(self, aggressive: bool = False):
        """Enhanced memory cleanup with multiple strategies"""
        self.logger.info(f"Performing {'aggressive' if aggressive else 'standard'} memory cleanup...")
        
        # Store memory before cleanup for comparison
        memory_before = self._get_memory_usage()
        self._last_memory_before_cleanup = memory_before["total_memory_gb"]
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if aggressive:
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
        
        # Force garbage collection
        gc.collect()
        
        # Always clear model cache after each generation
        if hasattr(self, '_MODEL_CACHE') and self._MODEL_CACHE:
            self.logger.info("Clearing model cache after generation")
            self._MODEL_CACHE.clear()
        
        # Clear any cached tensors in the model
        if hasattr(self, 'model') and self.model is not None:
            try:
                # Clear model's internal cache
                if hasattr(self.model, 'clear_cache'):
                    self.model.clear_cache()
                # Clear any cached attention states
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
                    self.model.config.use_cache = False
                    # Re-enable after clearing
                    self.model.config.use_cache = True
            except Exception as e:
                self.logger.warning(f"Error clearing model cache: {e}")
        
        # Additional cleanup for aggressive mode
        if aggressive:
            # Force another garbage collection
            gc.collect()
            
            # Clear any cached tensors
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Check memory after cleanup
        memory_after = self._get_memory_usage()
        cleanup_reduction = memory_after["total_memory_gb"]
        
        # Calculate actual reduction
        reduction = self._last_memory_before_cleanup - cleanup_reduction
        
        self.logger.debug(f"Memory cleanup completed. Current usage: {cleanup_reduction:.2f}GB")
        self.logger.info(f"Memory reduction: {reduction:.2f}GB")
        
        return memory_after

    def get_current_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics for monitoring"""
        memory_usage = self._get_memory_usage()
        return {
            'total_memory_gb': memory_usage['total_memory_gb'],
            'cpu_memory_gb': memory_usage['cpu_memory_gb'],
            'gpu_memory_gb': memory_usage['gpu_memory_gb'],
            'available_system_gb': memory_usage['available_system_gb'],
            'system_memory_percent': memory_usage['system_memory_percent'],
            'max_memory_limit_gb': self.max_memory_usage_gb,
            'memory_usage_percent': (memory_usage['total_memory_gb'] / self.max_memory_usage_gb) * 100,
            'memory_cleanup_enabled': self.enable_memory_cleanup
        }

    def format_prompt(self, raw_prompt: str) -> str:
        return (
            self.prompt_format
            .replace("{{user_prefix}}", self.user_prefix)
            .replace("{{assistant_prefix}}", self.assistant_prefix)
            .replace("{{prompt}}", raw_prompt)
        )

    def generate_response(self, prompt: str, task_type: str = "mutation_crossover") -> str:
        """Generate a single response for a prompt."""
        import time
        start_time = time.time()
        
        try:
            # Get task template
            template = self.task_templates.get(task_type, "")
            if not template:
                self.logger.warning(f"No template found for task type: {task_type}")
                return ""
            
            # Format prompt with template
            try:
                formatted_prompt = template.format(prompt=prompt)
            except KeyError:
                # Template doesn't have {prompt} placeholder, use prompt as-is
                formatted_prompt = prompt
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Get generation args
            task_args = self.task_generation_args.get(task_type, {})
            generation_kwargs = {
                **self.generation_args,
                **task_args,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            # Generate response
            with torch.no_grad():
                try:
                    if self.device in ("cuda", "mps"):
                        with torch.autocast(device_type=self.device, enabled=True):
                            outputs = self.model.generate(**inputs, **generation_kwargs)
                    else:
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                except Exception as e:
                    self.logger.warning(f"GPU generation failed, falling back to CPU: {e}")
                    inputs_cpu = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in inputs.items()}
                    outputs = self.model.to("cpu").generate(**inputs_cpu, **generation_kwargs)
            
            # Decode response
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = decoded[len(formatted_prompt):].strip() if decoded.startswith(formatted_prompt) else decoded.strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}", exc_info=True)
            return ""
        finally:
            end_time = time.time()
            generation_time = end_time - start_time
            # Store timing in a global variable that can be accessed by the caller
            if not hasattr(self, '_last_generation_time'):
                self._last_generation_time = {}
            self._last_generation_time['start_time'] = start_time
            self._last_generation_time['end_time'] = end_time
            self._last_generation_time['duration'] = generation_time

    def process_population(self, pop_path: str = "outputs/Population.json") -> None:
        """Process entire population for text generation one genome at a time"""
        import time
        
        try:
            self.logger.info("Starting population processing for text generation")
            
            # Load population
            population = self._load_population(pop_path)
            
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
                    response = self.generate_response(genome['prompt'], task_type="mutation_crossover")
                    
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
            
            # No need to save population here - each genome is saved immediately
            
            # Log final summary
            self.logger.info("Population processing completed:")
            self.logger.info("  - Total genomes: %d", len(population))
            self.logger.info("  - Processed: %d", total_processed)
            self.logger.info("  - Errors: %d", total_errors)
            
        except Exception as e:
            self.logger.error("Population processing failed: %s", e, exc_info=True)
            raise



    def _load_population(self, pop_path: str) -> List[Dict[str, Any]]:
        from utils.population_io import load_population
        return load_population(pop_path, logger=self.logger)
    
    def _save_population(self, population: List[Dict[str, Any]], pop_path: str) -> None:
        from utils.population_io import save_population
        save_population(population, pop_path, logger=self.logger)
    
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


    
    def _process_genome(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single genome with comprehensive logging"""
        genome_id = genome.get('id', 'unknown')
        
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
            
            # Use real model generation
            self.logger.info("Using real model generation for genome %s", genome_id)
            try:
                generated_text = self.generate_response(prompt)
                
                # Add timing information to genome
                if hasattr(self, '_last_generation_time'):
                    genome['generation_timing'] = self._last_generation_time.copy()
                
                # Update performance metrics for real generation
                self.generation_count += 1
                self.total_tokens_generated += len(generated_text.split())
                
                # Update genome
                genome['generated_output'] = generated_text
                genome['status'] = 'pending_evaluation'
                genome['model_name'] = self.model_cfg.get("name", "")
                
                self.logger.info("Successfully generated text for genome %s: %d characters", 
                               genome_id, len(generated_text))
                
                return genome
                
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
    
    def generate_raw_response(self, prompt: str, generation_kwargs: Dict[str, Any] = None) -> str:
        """
        Generate raw model response without any template manipulation or post-processing.
        
        This method gives you the pure model response exactly as the model generated it,
        without any filtering, cleaning, or template formatting.
        
        Args:
            prompt (str): Raw prompt to send to the model
            generation_kwargs (Dict[str, Any], optional): Custom generation parameters
            
        Returns:
            str: Raw model response without any manipulation
            
        Example:
            >>> generator = LlaMaTextGenerator(config_path="config/modelConfig.yaml")
            >>> raw_response = generator.generate_raw_response("What is artificial intelligence?")
            >>> print(raw_response)  # Pure model response
        """
        try:
            if generation_kwargs is None:
                generation_kwargs = self.generation_args.copy()
            
            # Tokenize the raw prompt directly (no template formatting)
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate with custom args
            generation_kwargs.update({
                "pad_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            })
            
            with torch.no_grad():
                if self.device in ("cuda", "mps"):
                    with torch.autocast(device_type=self.device, enabled=True):
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                else:
                    outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # Decode the FULL response (including the input prompt)
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the input prompt)
            if full_response.startswith(prompt):
                raw_response = full_response[len(prompt):].strip()
            else:
                raw_response = full_response.strip()
            
            self.logger.debug(f"Raw response for '{prompt[:50]}...': '{raw_response[:100]}...'")
            return raw_response
            
        except Exception as e:
            self.logger.error(f"Raw response generation failed: {e}")
            return f"Error: {e}"

    def translate(self, text: str, target_language: str, source_language: str = "English") -> str:
        """High-precision translation using task templates with robust tag extraction.

        - Uses deterministic generation args from config (task_generation_args.translation).
        - Extracts exactly the content inside <trans>...</trans> (case-insensitive).
        - Falls back conservatively if the model fails to follow the format.
        """
        try:
            # 1) Pick the right template
            templates = self.task_templates.get("translation", {})
            if str(source_language).strip().lower() == "english":
                template = templates.get("en_to_target")
            else:
                template = templates.get("target_to_en")

            if not template:
                # If templates are missing, return original text rather than crashing
                self.logger.warning("Translation template missing; returning original text.")
                return text

            prompt = template.format(
                text=text,
                target_language=target_language,
                source_language=source_language
            )

            # 2) Format with global prompt wrapper (keeps current project behavior)
            formatted_prompt = self.format_prompt(prompt)

            # 3) Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # 4) Merge generation args (global + task-specific)
            # Use mutation_crossover args for all operators including translation
            task_args = self.task_generation_args.get("mutation_crossover", {})
            generation_kwargs = {
                **self.generation_args,
                **task_args,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            # 5) Generate with device-specific optimizations
            with torch.no_grad():
                try:
                    if self.device in ("cuda", "mps"):
                        # Use autocast for GPU acceleration
                        with torch.autocast(device_type=self.device, enabled=True):
                            outputs = self.model.generate(**inputs, **generation_kwargs)
                    else:
                        # CPU generation without autocast
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                except Exception as e:
                    self.logger.warning(f"GPU generation failed, falling back to CPU: {e}")
                    # Fallback to CPU if GPU generation fails
                    inputs_cpu = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in inputs.items()}
                    outputs = self.model.to("cpu").generate(**inputs_cpu, **generation_kwargs)

            # 6) Decode full text and strip the prompt echo
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = decoded[len(formatted_prompt):].strip() if decoded.startswith(formatted_prompt) else decoded.strip()

            # Extract translation from structured tags
            import re
            m = re.search(r"<\s*trans\s*>([\s\S]*?)<\s*/\s*trans\s*>", response, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                # Clean common artifacts like surrounding quotes
                if len(candidate) >= 2 and (
                    (candidate[0] == candidate[-1] == '"') or
                    (candidate[0] == candidate[-1] == "'")
                ):
                    candidate = candidate[1:-1].strip()
                return candidate if candidate else text
            
            # If no structured tags, return original text
            return text

        except Exception as e:
            self.logger.error(f"Translation failed: {e}", exc_info=True)
            return text

    def paraphrase(self, text: str, north_star_metric: str, generated_output: str = "", current_score: float = 0.0) -> str:
        """High-precision paraphrasing using task templates with robust extraction.

        - Uses generation args from config (task_generation_args.mutation_crossover).
        - Extracts paraphrased content from structured response.
        - Falls back conservatively if the model fails to follow the format.
        """
        try:
            # 1) Get the paraphrasing template
            template = self.task_templates.get("paraphrasing", "")

            if not template:
                # If templates are missing, return original text rather than crashing
                self.logger.warning("Paraphrasing template missing; returning original text.")
                return text

            prompt = template.format(
                north_star_metric=north_star_metric,
                original_prompt=text,
                generated_output=generated_output,
                current_score=current_score
            )

            # 2) Format with global prompt wrapper (keeps current project behavior)
            formatted_prompt = self.format_prompt(prompt)

            # 3) Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # 4) Merge generation args (global + task-specific)
            task_args = self.task_generation_args.get("mutation_crossover", {})
            generation_kwargs = {
                **self.generation_args,
                **task_args,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            # 5) Generate with device-specific optimizations
            with torch.no_grad():
                try:
                    if self.device in ("cuda", "mps"):
                        # Use autocast for GPU acceleration
                        with torch.autocast(device_type=self.device, enabled=True):
                            outputs = self.model.generate(**inputs, **generation_kwargs)
                    else:
                        # CPU generation without autocast
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                except Exception as e:
                    self.logger.warning(f"GPU generation failed, falling back to CPU: {e}")
                    # Fallback to CPU if GPU generation fails
                    inputs_cpu = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in inputs.items()}
                    outputs = self.model.to("cpu").generate(**inputs_cpu, **generation_kwargs)

            # 6) Decode full text and strip the prompt echo
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = decoded[len(formatted_prompt):].strip() if decoded.startswith(formatted_prompt) else decoded.strip()

            # Extract paraphrased content from structured tags
            import re
            
            # First try to extract from structured tags
            paraphrase_match = re.search(r'<paraphrase>(.*?)</paraphrase>', response, re.DOTALL)
            if paraphrase_match:
                candidate = paraphrase_match.group(1).strip()
                if candidate and candidate.lower() != text.lower():
                    return candidate
            
            # Fallback: Extract question from response
            candidate = self._extract_question_from_response(response)
            if candidate and candidate.lower() != text.lower():
                return candidate
            
            # If no structured tags, return original text
            return text

        except Exception as e:
            self.logger.error(f"Paraphrasing failed: {e}", exc_info=True)
            return text

    def stylistic_mutate(self, text: str, style_attribute: str) -> str:
        """High-precision stylistic mutation using task templates with robust extraction.

        - Uses generation args from config (task_generation_args.mutation_crossover).
        - Extracts stylistically modified content from structured response.
        - Falls back conservatively if the model fails to follow the format.
        """
        try:
            # 1) Get the stylistic mutation template
            template = self.task_templates.get("stylistic_mutation", "")

            if not template:
                # If templates are missing, return original text rather than crashing
                self.logger.warning("Stylistic mutation template missing; returning original text.")
                return text

            prompt = template.format(
                original_text=text,
                style_attribute=style_attribute
            )

            # 2) Format with global prompt wrapper (keeps current project behavior)
            formatted_prompt = self.format_prompt(prompt)

            # 3) Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # 4) Merge generation args (global + task-specific)
            task_args = self.task_generation_args.get("mutation_crossover", {})
            generation_kwargs = {
                **self.generation_args,
                **task_args,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            # 5) Generate with device-specific optimizations
            with torch.no_grad():
                try:
                    if self.device in ("cuda", "mps"):
                        # Use autocast for GPU acceleration
                        with torch.autocast(device_type=self.device, enabled=True):
                            outputs = self.model.generate(**inputs, **generation_kwargs)
                    else:
                        # CPU generation without autocast
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                except Exception as e:
                    self.logger.warning(f"GPU generation failed, falling back to CPU: {e}")
                    # Fallback to CPU if GPU generation fails
                    inputs_cpu = {k: v.to("cpu") if hasattr(v, 'to') else v for k, v in inputs.items()}
                    outputs = self.model.to("cpu").generate(**inputs_cpu, **generation_kwargs)

            # 6) Decode full text and strip the prompt echo
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = decoded[len(formatted_prompt):].strip() if decoded.startswith(formatted_prompt) else decoded.strip()

            # Extract stylistically modified content from structured tags
            import re
            
            # First try to extract from structured tags
            modified_match = re.search(r'<modified>(.*?)</modified>', response, re.DOTALL)
            if modified_match:
                candidate = modified_match.group(1).strip()
                if candidate and candidate.lower() != text.lower():
                    return candidate
            
            # Fallback: Extract question from response
            candidate = self._extract_question_from_response(response)
            if candidate and candidate.lower() != text.lower():
                return candidate
            
            # If no structured tags, return original text
            return text

        except Exception as e:
            self.logger.error(f"Stylistic mutation failed: {e}", exc_info=True)
            return text
    
    def _extract_question_from_response(self, response: str) -> str:
        """Extract a question from LLM response as fallback parsing."""
        try:
            import re
            # Look for sentences ending with question marks
            questions = re.findall(r'[^.!?]*\?', response)
            if questions:
                for question in questions:
                    question = question.strip()
                    if len(question) > 10 and question.endswith('?'):
                        return question
            
            # Look for sentences that start with question words
            question_patterns = [
                r'(?:How|What|Why|When|Where|Who|Which|Can|Could|Should|Would|Do|Does|Did|Is|Are|Was|Were|Will|Shall)\s+[^.!?]*[.!?]',
                r'(?:How|What|Why|When|Where|Who|Which|Can|Could|Should|Would|Do|Does|Did|Is|Are|Was|Were|Will|Shall)\s+[^.!?]*\?'
            ]
            
            for pattern in question_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    for match in matches:
                        match = match.strip()
                        if len(match) > 10:
                            return match
            
            # Look for any sentence that could be a question
            sentences = re.split(r'[.!?]+', response)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and any(word in sentence.lower() for word in ['how', 'what', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'should', 'would']):
                    return sentence + '?'
            
            return ""
            
        except Exception as e:
            self.logger.debug(f"Failed to extract question from response: {e}")
            return ""



