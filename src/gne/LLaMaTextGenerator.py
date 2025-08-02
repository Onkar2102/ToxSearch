import os
import json
import torch
import yaml
import gc
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.custom_logging import get_logger, PerformanceLogger
from typing import List, Dict, Any, Optional
import time
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

        # Memory management settings
        self.enable_memory_cleanup = self.model_cfg.get("enable_memory_cleanup", True)
        self.max_memory_usage_gb = self.model_cfg.get("max_memory_usage_gb", 4.0)
        
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
        
        # Memory monitoring
        self.last_memory_check = time.time()
        self.memory_check_interval = 60  # Check memory every 60 seconds

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
        
        # Enhanced memory threshold checking
        memory_warning_threshold = self.max_memory_usage_gb * 0.8  # Warning at 80%
        memory_critical_threshold = self.max_memory_usage_gb * 0.95  # Critical at 95%
        
        # Check system memory availability
        if memory_usage["available_system_gb"] < 1.0:  # Less than 1GB available
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
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if aggressive:
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
        
        # Force garbage collection
        gc.collect()
        
        # Additional cleanup for aggressive mode
        if aggressive:
            # Clear model cache
            if hasattr(self, '_MODEL_CACHE') and self._MODEL_CACHE:
                self.logger.info("Clearing model cache for aggressive cleanup")
                self._MODEL_CACHE.clear()
            
            # Force another garbage collection
            gc.collect()
            
            # Clear any cached tensors
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Check memory after cleanup
        memory_usage = self._get_memory_usage()
        cleanup_reduction = memory_usage["total_memory_gb"]
        
        self.logger.info(f"Memory cleanup completed. Current usage: {cleanup_reduction:.2f}GB")
        
        # Log cleanup effectiveness
        if hasattr(self, '_last_memory_before_cleanup'):
            reduction = self._last_memory_before_cleanup - cleanup_reduction
            self.logger.info(f"Memory reduction: {reduction:.2f}GB")
        
        return memory_usage

    def _adaptive_batch_size(self, prompts: List[str]) -> int:
        """Enhanced adaptive batch sizing based on real-time memory monitoring"""
        memory_usage = self._get_memory_usage()
        
        # Calculate available memory
        available_memory = self.max_memory_usage_gb - memory_usage["total_memory_gb"]
        system_available = memory_usage["available_system_gb"]
        
        # Use the more conservative available memory
        effective_available = min(available_memory, system_available)
        
        # Estimate memory per prompt based on model size and sequence length
        max_tokens = self.generation_args.get("max_new_tokens", 1024)
        estimated_memory_per_prompt = (max_tokens * 0.0001) + 0.1  # Rough estimate
        
        # Calculate optimal batch size
        if effective_available < 0.5:  # Critical memory situation
            self.logger.warning(f"Critical memory situation: {effective_available:.2f}GB available")
            return 1
        elif effective_available < 1.0:  # Low memory
            optimal_batch = max(1, int(effective_available / estimated_memory_per_prompt))
            return min(optimal_batch, len(prompts), 2)
        elif effective_available < 2.0:  # Moderate memory
            optimal_batch = max(1, int(effective_available / estimated_memory_per_prompt))
            return min(optimal_batch, len(prompts), self.max_batch_size)
        else:  # Good memory availability
            optimal_batch = int(effective_available / estimated_memory_per_prompt)
            return min(optimal_batch, len(prompts), self.max_batch_size)

    def format_prompt(self, raw_prompt: str) -> str:
        return (
            self.prompt_format
            .replace("{{user_prefix}}", self.user_prefix)
            .replace("{{assistant_prefix}}", self.assistant_prefix)
            .replace("{{prompt}}", raw_prompt)
        )

    def generate_response_batch(self, prompts: List[str]) -> List[str]:
        """Enhanced batch generation with real-time memory monitoring and adaptive sizing"""
        if not prompts:
            return []
        
        # Store initial memory state for tracking
        initial_memory = self._get_memory_usage()
        self.logger.info(f"Starting batch generation. Initial memory: {initial_memory['total_memory_gb']:.2f}GB")
        
        # Check memory and cleanup if necessary
        if not self._check_memory_and_cleanup():
            self.logger.error("Memory usage too high, cannot proceed with generation")
            return ["[MEMORY_ERROR]" for _ in prompts]
        
        # Use enhanced adaptive batch size
        adaptive_batch_size = self._adaptive_batch_size(prompts)
        if adaptive_batch_size < len(prompts):
            self.logger.info(f"Adaptive batch sizing: reduced from {len(prompts)} to {adaptive_batch_size} prompts")
        
        # Process in smaller batches with enhanced monitoring
        all_responses = []
        batch_count = 0
        
        for i in range(0, len(prompts), adaptive_batch_size):
            batch_count += 1
            batch_prompts = prompts[i:i + adaptive_batch_size]
            
            # Log batch start
            self.logger.debug(f"Processing batch {batch_count}/{len(prompts)//adaptive_batch_size + 1} with {len(batch_prompts)} prompts")
            
            # Store memory before batch
            self._last_memory_before_cleanup = self._get_memory_usage()["total_memory_gb"]
            
            try:
                batch_responses = self._generate_single_batch(batch_prompts)
                all_responses.extend(batch_responses)
                
                # Log batch completion
                current_memory = self._get_memory_usage()
                memory_increase = current_memory["total_memory_gb"] - self._last_memory_before_cleanup
                self.logger.debug(f"Batch {batch_count} completed. Memory change: {memory_increase:+.2f}GB")
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_count}: {e}")
                # Add error responses for failed batch
                all_responses.extend([f"[GENERATION_ERROR: {str(e)}]" for _ in batch_prompts])
            
            # Enhanced cleanup after each batch
            if self.enable_memory_cleanup:
                cleanup_result = self._check_memory_and_cleanup(force=True)
                if not cleanup_result:
                    self.logger.warning(f"Memory cleanup failed after batch {batch_count}")
        
        # Final memory report
        final_memory = self._get_memory_usage()
        total_memory_change = final_memory["total_memory_gb"] - initial_memory["total_memory_gb"]
        self.logger.info(f"Batch generation completed. Total memory change: {total_memory_change:+.2f}GB")
        
        return all_responses

    def _generate_single_batch(self, prompts: List[str]) -> List[str]:
        """Enhanced single batch generation with comprehensive error handling"""
        formatted_prompts = [self.format_prompt(prompt) for prompt in prompts]
        
        # Pre-generation memory check
        pre_gen_memory = self._get_memory_usage()
        self.logger.debug(f"Pre-generation memory: {pre_gen_memory['total_memory_gb']:.2f}GB")
        
        try:
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
            
            # Post-generation memory check
            post_gen_memory = self._get_memory_usage()
            memory_used = post_gen_memory["total_memory_gb"] - pre_gen_memory["total_memory_gb"]
            self.logger.debug(f"Generation completed. Memory used: {memory_used:+.2f}GB")
            
            return responses
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.error(f"Out of memory error during generation: {e}")
                # Aggressive cleanup after OOM
                self._cleanup_memory(aggressive=True)
                return ["[OOM_ERROR]" for _ in prompts]
            elif "cuda" in str(e).lower() and "memory" in str(e).lower():
                self.logger.error(f"CUDA memory error: {e}")
                self._cleanup_memory(aggressive=True)
                return ["[CUDA_MEMORY_ERROR]" for _ in prompts]
            else:
                self.logger.error(f"Runtime error during generation: {e}")
                return ["[RUNTIME_ERROR]" for _ in prompts]
                
        except Exception as e:
            self.logger.error(f"Unexpected error during generation: {e}")
            return ["[GENERATION_ERROR]" for _ in prompts]

    def generate_response(self, prompt: str) -> str:
        """Single prompt generation (backwards compatibility)"""
        return self.generate_response_batch([prompt])[0]

    def process_population(self, pop_path: str = "outputs/Population.json", batch_size: int = None) -> None:
        """Process entire population for text generation with batch saving for fault tolerance"""
        # Use config batch size if not provided, fallback to default
        batch_size = batch_size or self.model_cfg.get("generation_batch_size", 10)
        
        with PerformanceLogger(self.logger, "Process Population", pop_path=pop_path, batch_size=batch_size):
            try:
                self.logger.info("Starting population processing for text generation with batch saving")
                self.logger.info("Using batch size: %d (from config: %s)", batch_size, self.model_cfg.get("generation_batch_size", "default"))
                
                # Load population
                population = self._load_population(pop_path)
                
                # Count genomes that need processing
                pending_genomes = [g for g in population if g.get('status') == 'pending_generation']
                self.logger.info("Found %d genomes pending generation out of %d total", len(pending_genomes), len(population))
                
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



    def _load_population(self, pop_path: str) -> List[Dict[str, Any]]:
        return load_population(pop_path, logger=self.logger)

    def _save_population(self, population: List[Dict[str, Any]], pop_path: str) -> None:
        save_population(population, pop_path, logger=self.logger)


    
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
    

    
