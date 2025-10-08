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
get_logger, _, _, PerformanceLogger = get_custom_logging()

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
        self.adaptive_batch_sizing = self.model_cfg.get("adaptive_batch_sizing", True)
        self.min_batch_size = self.model_cfg.get("min_batch_size", 1)
        self.max_batch_size_memory = self.model_cfg.get("max_batch_size_memory", 4)

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
        self.max_batch_size = self.model_cfg.get("max_batch_size", 4)
        self.logger.info(f"Model loaded on {self.device} with batch size {self.max_batch_size}")

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
        
        self.logger.info(f"Memory cleanup completed. Current usage: {cleanup_reduction:.2f}GB")
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
            'memory_cleanup_enabled': self.enable_memory_cleanup,
            'adaptive_batch_sizing': self.adaptive_batch_sizing
        }

    def _adaptive_batch_size(self, prompts: List[str]) -> int:
        """Enhanced adaptive batch sizing based on real-time memory monitoring"""
        if not self.adaptive_batch_sizing:
            return min(self.max_batch_size, len(prompts))
        
        memory_usage = self._get_memory_usage()
        
        # Calculate available memory
        available_memory = self.max_memory_usage_gb - memory_usage["total_memory_gb"]
        system_available = memory_usage["available_system_gb"]
        
        # Use the more conservative available memory
        effective_available = min(available_memory, system_available)
        
        # Calculate memory usage percentage
        memory_usage_percent = (memory_usage["total_memory_gb"] / self.max_memory_usage_gb) * 100
        
        # Estimate memory per prompt based on model size and sequence length
        max_tokens = self.generation_args.get("max_new_tokens", 4096)
        estimated_memory_per_prompt = (max_tokens * 0.0001) + 0.2  # More conservative estimate
        
        # Dynamic batch sizing based on memory pressure
        if memory_usage_percent > 90:  # Critical memory situation
            self.logger.warning(f"Critical memory situation: {memory_usage_percent:.1f}% used, {effective_available:.2f}GB available")
            return self.min_batch_size
        elif memory_usage_percent > 80:  # High memory pressure
            self.logger.warning(f"High memory pressure: {memory_usage_percent:.1f}% used")
            optimal_batch = max(self.min_batch_size, int(effective_available / estimated_memory_per_prompt))
            return min(optimal_batch, len(prompts), 2)
        elif memory_usage_percent > 70:  # Moderate memory pressure
            self.logger.info(f"Moderate memory pressure: {memory_usage_percent:.1f}% used")
            optimal_batch = max(self.min_batch_size, int(effective_available / estimated_memory_per_prompt))
            return min(optimal_batch, len(prompts), self.max_batch_size_memory)
        elif memory_usage_percent > 50:  # Good memory availability
            self.logger.info(f"Good memory availability: {memory_usage_percent:.1f}% used")
            optimal_batch = int(effective_available / estimated_memory_per_prompt)
            return min(optimal_batch, len(prompts), self.max_batch_size_memory)
        else:  # Excellent memory availability
            self.logger.info(f"Excellent memory availability: {memory_usage_percent:.1f}% used")
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
        self._check_memory_usage()
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
        
        # Force memory cleanup after all batches are processed
        if self.enable_memory_cleanup:
            self.logger.info("Performing final memory cleanup after generation")
            self._cleanup_memory(aggressive=True)
        
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
                if 'System:' in response:
                    response = response.split('System:')[-1].strip()
                
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
            elif "probability tensor" in str(e).lower() or "inf" in str(e).lower() or "nan" in str(e).lower():
                self.logger.error(f"Probability tensor error during generation: {e}")
                # Try with safer generation parameters
                try:
                    safe_kwargs = generation_kwargs.copy()
                    safe_kwargs.update({
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repetition_penalty": 1.1
                    })
                    self.logger.info("Retrying with safer generation parameters...")
                    with torch.no_grad():
                        if self.device in ("cuda", "mps"):
                            with torch.autocast(device_type=self.device, enabled=True):
                                outputs = self.model.generate(**inputs, **safe_kwargs)
                        else:
                            outputs = self.model.generate(**inputs, **safe_kwargs)
                    
                    responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                    return responses
                except Exception as retry_e:
                    self.logger.error(f"Retry with safe parameters also failed: {retry_e}")
                    return ["[PROBABILITY_TENSOR_ERROR]" for _ in prompts]
            else:
                self.logger.error(f"Runtime error during generation: {e}")
                return ["[RUNTIME_ERROR]" for _ in prompts]
                
        except Exception as e:
            self.logger.error(f"Unexpected error during generation: {e}")
            return ["[GENERATION_ERROR]" for _ in prompts]

    def generate_response(self, prompt: str, task_type: str = None) -> str:
        """
        Single prompt generation with optional task-specific parameters.
        
        Args:
            prompt (str): The prompt to generate a response for
            task_type (str, optional): Task type to use specific generation parameters
            
        Returns:
            str: Generated response
        """
        if task_type and task_type in self.task_generation_args:
            # Use task-specific generation arguments
            task_args = self.task_generation_args[task_type].copy()
            # Merge with base generation args, task args take precedence
            generation_kwargs = self.generation_args.copy()
            generation_kwargs.update(task_args)
            
            # Generate with task-specific parameters
            return self.generate_raw_response(prompt, generation_kwargs)
        else:
            # Use default generation arguments
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
        from utils.population_io import load_population
        return load_population(pop_path, logger=self.logger)
    
    def _save_population(self, population: List[Dict[str, Any]], pop_path: str) -> None:
        from utils.population_io import save_population
        save_population(population, pop_path, logger=self.logger)


    
    def _process_genome(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single genome with comprehensive logging"""
        genome_id = genome.get('id', 'unknown')
        
        get_logger, _, _, PerformanceLogger = get_custom_logging()
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

            # 7) Extract <trans>...</trans> (robust, single capture)
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

                # Final guardrails: if the model echoed the source unchanged during back-translation,
                # still return it (caller may decide). Otherwise, prefer non-empty candidate.
                return candidate if candidate else text

            # 8) If tag not found, attempt a minimal fallback:
            #    - remove any leading role-like lines the model might have added
            #    - then return the first non-empty line
            for line in response.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Skip obvious template echoes
                if line.lower().startswith(("system:", "user:", "assistant:")):
                    continue
                # If the model returned raw translation without tags, accept it.
                return line

            # 9) Ultimate fallback: original text (never crash)
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
            templates = self.task_templates.get("paraphrasing", {})
            template = templates.get("instruction_preserving")

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

            # 7) Extract paraphrased content using multiple patterns
            import re
            patterns = [
                r"Paraphrased Prompt:\s*(.+?)(?:\n\n|\n$|$)",
                r"Paraphrased:\s*(.+?)(?:\n\n|\n$|$)",
                r"Variant 1:\s*(.+?)(?:\n\n|\n$|$)",
                r"1\.\s*(.+?)(?:\n\n|\n$|$)"
            ]
            
            for pattern in patterns:
                m = re.search(pattern, response, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    candidate = m.group(1).strip()
                    # Clean common artifacts like surrounding quotes
                    if len(candidate) >= 2 and (
                        (candidate[0] == candidate[-1] == '"') or
                        (candidate[0] == candidate[-1] == "'")
                    ):
                        candidate = candidate[1:-1].strip()
                    if candidate and candidate.lower() != text.lower():
                        return candidate

            # 8) Fallback: look for quoted content
            quoted_content = re.findall(r'"([^"]+)"', response)
            for quote in quoted_content:
                if quote.strip() and quote.lower() != text.lower():
                    return quote.strip()

            # 9) Fallback: look for first substantial line
            for line in response.splitlines():
                line = line.strip()
                if not line or len(line) < 10:
                    continue
                if line.lower().startswith(("system:", "user:", "assistant:", "paraphrased prompt:", "instructions:")):
                    continue
                if line.lower() != text.lower():
                    return line

            # 10) Ultimate fallback: original text (never crash)
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
            templates = self.task_templates.get("stylistic_mutation", {})
            template = templates.get("style_modification")

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

            # 7) Extract stylistically modified content using multiple patterns
            import re
            patterns = [
                r"Modified Text:\s*(.+?)(?:\n\n|\n$|$)",
                r"Stylistic Variant:\s*(.+?)(?:\n\n|\n$|$)",
                r"Result:\s*(.+?)(?:\n\n|\n$|$)",
                r"Output:\s*(.+?)(?:\n\n|\n$|$)"
            ]
            
            for pattern in patterns:
                m = re.search(pattern, response, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    candidate = m.group(1).strip()
                    # Clean common artifacts like surrounding quotes
                    if len(candidate) >= 2 and (
                        (candidate[0] == candidate[-1] == '"') or
                        (candidate[0] == candidate[-1] == "'")
                    ):
                        candidate = candidate[1:-1].strip()
                    if candidate and candidate.lower() != text.lower():
                        return candidate

            # 8) Fallback: look for quoted content
            quoted_content = re.findall(r'"([^"]+)"', response)
            for quote in quoted_content:
                if quote.strip() and quote.lower() != text.lower():
                    return quote.strip()

            # 9) Fallback: look for first substantial line
            for line in response.splitlines():
                line = line.strip()
                if not line or len(line) < 10:
                    continue
                if line.lower().startswith(("system:", "user:", "assistant:", "modified text:", "instructions:")):
                    continue
                if line.lower() != text.lower():
                    return line

            # 10) Ultimate fallback: original text (never crash)
            return text

        except Exception as e:
            self.logger.error(f"Stylistic mutation failed: {e}", exc_info=True)
            return text



