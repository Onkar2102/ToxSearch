"""
Model Interface Abstraction Layer

Provides OpenAI-compatible v1/chat/completions interface for model-agnostic architecture.
Currently implements llama_cpp provider with chat completions support.
"""

import os
import time
import psutil
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from utils import get_custom_logging

# Get the functions at module level to avoid repeated calls
get_logger, _, _, _ = get_custom_logging()


class ModelInterface(ABC):
    """Abstract base class for model interfaces."""
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat completion response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass


class LlamaCppChatInterface(ModelInterface):
    """LlamaCpp implementation of chat completions interface."""
    
    _MODEL_CACHE = {}
    _MODEL_CACHE_ACCESS_COUNT = {}  # Track access frequency
    _MODEL_CACHE_LOCK = None  # Thread safety
    
    def __init__(self, model_cfg: Dict[str, Any], log_file: Optional[str] = None):
        """
        Initialize the LlamaCpp chat interface.
        
        Args:
            model_cfg: Model configuration dictionary
            log_file: Optional log file path
        """
        self.logger = get_logger("LlamaCppChatInterface", log_file)
        self.model_cfg = model_cfg
        
        # Memory management settings
        self.enable_memory_cleanup = model_cfg.get("enable_memory_cleanup", True)
        self.max_memory_usage_gb = model_cfg.get("max_memory_usage_gb", 12.0)
        
        # Performance tracking
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        
        # Load model
        model_path = model_cfg["name"]
        
        # Convert relative paths to absolute paths for consistent caching
        if not os.path.isabs(model_path):
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[2]
            absolute_model_path = str(project_root / model_path)
        else:
            absolute_model_path = model_path
        
        # Thread-safe model loading
        import threading
        if self._MODEL_CACHE_LOCK is None:
            self._MODEL_CACHE_LOCK = threading.Lock()
        
        with self._MODEL_CACHE_LOCK:
            # Use absolute path for caching
            if absolute_model_path not in self._MODEL_CACHE:
                self.logger.info(f"Loading llama.cpp model: {absolute_model_path}")
                self._load_model(absolute_model_path)
                # Track access count for new model
                self._MODEL_CACHE_ACCESS_COUNT[absolute_model_path] = 1
                self.logger.info(f"Model loaded and cached: {absolute_model_path}")
            else:
                self.logger.info(f"Using cached llama.cpp model: {absolute_model_path}")
                # Increment access count
                self._MODEL_CACHE_ACCESS_COUNT[absolute_model_path] = self._MODEL_CACHE_ACCESS_COUNT.get(absolute_model_path, 0) + 1
        
        # Cleanup cache if too many models (but preserve our two main models)
        self._cleanup_model_cache_if_needed()
        
        self.model = self._MODEL_CACHE[absolute_model_path]
        self.generation_args = model_cfg.get("generation_args", {})
        
        # Memory monitoring
        self.last_memory_check = time.time()
        self.memory_check_interval = 60  # Check memory every 60 seconds
    
    def _cleanup_model_cache_if_needed(self):
        """Clean up unused models from cache, but preserve main RG/PG models."""
        max_cache_size = 5  # Allow some extra models but keep it reasonable
        
        if len(self._MODEL_CACHE) > max_cache_size:
            # Identify main models (RG and PG) to preserve
            main_models = set()
            for model_path in self._MODEL_CACHE.keys():
                if any(keyword in model_path.lower() for keyword in ['q4_k_m', 'q3_k_s']):
                    main_models.add(model_path)
            
            # Remove least used models, but preserve main models
            models_to_remove = []
            for model_path, access_count in self._MODEL_CACHE_ACCESS_COUNT.items():
                if model_path not in main_models:
                    models_to_remove.append((model_path, access_count))
            
            # Sort by access count and remove least used
            models_to_remove.sort(key=lambda x: x[1])
            excess_count = len(self._MODEL_CACHE) - max_cache_size
            
            for i in range(min(excess_count, len(models_to_remove))):
                model_path, access_count = models_to_remove[i]
                del self._MODEL_CACHE[model_path]
                del self._MODEL_CACHE_ACCESS_COUNT[model_path]
                self.logger.info(f"Removed model {model_path} from cache (access count: {access_count})")
    
    def _load_model(self, model_path: str):
        """Load model using llama.cpp with device-specific optimizations."""
        try:
            # Convert relative paths to absolute paths
            if not os.path.isabs(model_path):
                from pathlib import Path
                project_root = Path(__file__).resolve().parents[2]
                model_path = str(project_root / model_path)
            
            # Check if model file exists
            if not os.path.exists(model_path):
                # Try to find GGUF file
                gguf_path = f"{model_path}.gguf"
                if os.path.exists(gguf_path):
                    model_path = gguf_path
                else:
                    # Create a mock model for testing
                    self.logger.warning(f"Model file not found: {model_path}")
                    self.logger.info("Creating mock model for testing purposes")
                    mock_model = self._create_mock_model()
                    self._MODEL_CACHE[model_path] = mock_model
                    return
            
            # Get device-specific configuration
            device_config = self._get_device_specific_config()
            
            # Configure llama.cpp parameters with device-specific optimizations
            llama_params = {
                "model_path": model_path,
                "n_ctx": device_config.get("context_length", 4096),
                "n_threads": device_config.get("num_threads", None),
                "n_gpu_layers": device_config.get("gpu_layers", 0),
                "verbose": False,
                "use_mmap": device_config.get("use_mmap", True),
                "use_mlock": device_config.get("use_mlock", False),
                "low_vram": device_config.get("low_vram", False),
                "f16_kv": device_config.get("f16_kv", True),
                "logits_all": False,
                "vocab_only": False,
                "use_mmap": device_config.get("use_mmap", True),
                "use_mlock": device_config.get("use_mlock", False),
            }
            
            # Add device-specific parameters
            if device_config.get("device") == "mps":
                # Metal Performance Shaders optimizations for macOS
                llama_params.update({
                    "n_gpu_layers": device_config.get("gpu_layers", 20),  # Reasonable default for MPS
                    "main_gpu": 0,
                    "tensor_split": None,
                })
            elif device_config.get("device") == "cuda":
                # CUDA optimizations for NVIDIA GPUs
                llama_params.update({
                    "n_gpu_layers": device_config.get("gpu_layers", -1),  # Use all layers for CUDA
                    "main_gpu": 0,
                    "tensor_split": device_config.get("tensor_split", None),
                })
            else:
                # CPU optimizations
                llama_params.update({
                    "n_gpu_layers": 0,
                    "n_threads": device_config.get("num_threads", None),
                })
            
            # Load model
            self.logger.info(f"Loading model with llama.cpp on {device_config.get('device', 'cpu')}...")
            self.logger.debug(f"Llama.cpp parameters: {llama_params}")
            model = Llama(**llama_params)
            
            self._MODEL_CACHE[model_path] = model
            self.logger.info(f"Model loaded successfully: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Create mock model as fallback
            self.logger.info("Creating mock model as fallback")
            mock_model = self._create_mock_model()
            self._MODEL_CACHE[model_path] = mock_model
    
    def _get_device_specific_config(self) -> Dict[str, Any]:
        """Get device-specific configuration for llama.cpp."""
        from utils.device_utils import device_manager
        
        device = device_manager.get_optimal_device()
        config = self.model_cfg.get("device_config", {})
        
        # Base configuration
        device_config = {
            "device": device,
            "context_length": 4096,
            "num_threads": None,
            "use_mmap": True,
            "use_mlock": False,
            "low_vram": False,
            "f16_kv": True,
        }
        
        if device == "mps":
            # Metal Performance Shaders optimizations for macOS
            device_config.update({
                "gpu_layers": config.get("mps", {}).get("gpu_layers", -1),  # Reasonable default
                "use_mmap": True,
                "use_mlock": True,
                "low_vram": False,
                "f16_kv": True,
            })
        elif device == "cuda":
            # CUDA optimizations for NVIDIA GPUs
            cuda_config = config.get("cuda", {})
            device_config.update({
                "gpu_layers": cuda_config.get("gpu_layers", -1),  # Use all layers
                "use_mmap": True,
                "use_mlock": False,
                "low_vram": cuda_config.get("low_vram", False),
                "f16_kv": True,
                "tensor_split": cuda_config.get("tensor_split", None),
            })
        else:
            # CPU optimizations
            cpu_config = config.get("cpu", {})
            device_config.update({
                "gpu_layers": 0,
                "num_threads": cpu_config.get("num_threads", None),
                "use_mmap": True,
                "use_mlock": False,
                "low_vram": False,
                "f16_kv": False,  # Use f32 for CPU
            })
        
        return device_config
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get model cache statistics."""
        return {
            "cached_models": len(cls._MODEL_CACHE),
            "model_paths": list(cls._MODEL_CACHE.keys()),
            "access_counts": dict(cls._MODEL_CACHE_ACCESS_COUNT),
            "total_accesses": sum(cls._MODEL_CACHE_ACCESS_COUNT.values())
        }
    
    @classmethod
    def clear_cache(cls, preserve_main_models: bool = True):
        """Clear model cache, optionally preserving main RG/PG models."""
        if preserve_main_models:
            main_models = {}
            main_access_counts = {}
            for model_path in cls._MODEL_CACHE.keys():
                if any(keyword in model_path.lower() for keyword in ['q4_k_m', 'q3_k_s']):
                    main_models[model_path] = cls._MODEL_CACHE[model_path]
                    main_access_counts[model_path] = cls._MODEL_CACHE_ACCESS_COUNT.get(model_path, 0)
            
            cls._MODEL_CACHE.clear()
            cls._MODEL_CACHE_ACCESS_COUNT.clear()
            
            cls._MODEL_CACHE.update(main_models)
            cls._MODEL_CACHE_ACCESS_COUNT.update(main_access_counts)
        else:
            cls._MODEL_CACHE.clear()
            cls._MODEL_CACHE_ACCESS_COUNT.clear()
    
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
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to appropriate prompt format for GGUF models.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "").strip()
            
            if not content:
                continue
                
            if role == "system":
                # For GGUF models, system messages are typically handled differently
                # We'll prepend them to the user message
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                if prompt_parts and prompt_parts[-1].startswith("System:"):
                    # Combine system and user messages
                    prompt_parts[-1] += f"\n\nUser: {content}"
                else:
                    prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Join all parts with newlines
        formatted_prompt = "\n".join(prompt_parts)
        
        # Add a prompt for the model to continue
        if not formatted_prompt.endswith("Assistant:"):
            formatted_prompt += "\nAssistant:"
        
        return formatted_prompt
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text using a simple heuristic.
        This is a rough approximation - actual tokenization may vary.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Simple heuristic: ~4 characters per token for English text
        # This is conservative and may underestimate for some cases
        char_count = len(text)
        
        # Account for special tokens and whitespace
        # Count words as base, then add overhead for special tokens
        word_count = len(text.split())
        
        # Use the higher of character-based or word-based estimates
        char_based_estimate = char_count // 4
        word_based_estimate = word_count * 1.3  # ~1.3 tokens per word
        
        estimated_tokens = max(char_based_estimate, int(word_based_estimate))
        
        # Add overhead for special tokens (system prompts, formatting, etc.)
        overhead = 50  # Conservative overhead for special tokens
        
        return estimated_tokens + overhead
    
    def _validate_context_window(self, formatted_prompt: str, max_new_tokens: int, context_length: int = 4096) -> tuple[str, int]:
        """
        Validate and adjust prompt/tokens to fit within context window.
        
        Args:
            formatted_prompt: The formatted prompt text
            max_new_tokens: Requested maximum new tokens to generate
            context_length: Total context window length
            
        Returns:
            Tuple of (adjusted_prompt, adjusted_max_tokens)
        """
        prompt_tokens = self._estimate_token_count(formatted_prompt)
        total_requested = prompt_tokens + max_new_tokens
        
        self.logger.debug(f"Token validation: prompt={prompt_tokens}, max_new={max_new_tokens}, total={total_requested}, context={context_length}")
        
        if total_requested <= context_length:
            # Everything fits, return as-is
            return formatted_prompt, max_new_tokens
        
        # Need to adjust - prioritize keeping max_new_tokens if possible
        available_for_prompt = context_length - max_new_tokens
        
        if available_for_prompt <= 0:
            # Even max_new_tokens alone exceeds context window
            self.logger.warning(f"max_new_tokens ({max_new_tokens}) exceeds context window ({context_length}). Reducing to {context_length - 100}")
            return formatted_prompt, context_length - 100
        
        if prompt_tokens > available_for_prompt:
            # Need to truncate prompt
            self.logger.warning(f"Prompt too long ({prompt_tokens} tokens). Truncating to fit context window.")
            
            # Simple truncation strategy: keep the end of the prompt (usually user input)
            # This is a basic implementation - could be improved with smarter truncation
            truncated_prompt = self._truncate_prompt(formatted_prompt, available_for_prompt)
            truncated_tokens = self._estimate_token_count(truncated_prompt)
            
            self.logger.debug(f"Truncated prompt: {truncated_tokens} tokens")
            return truncated_prompt, max_new_tokens
        
        return formatted_prompt, max_new_tokens
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Truncate prompt to fit within token limit.
        Prioritizes keeping the end of the prompt (user input).
        
        Args:
            prompt: Original prompt text
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated prompt text
        """
        if not prompt:
            return prompt
        
        # Estimate characters per token (conservative)
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        
        if len(prompt) <= max_chars:
            return prompt
        
        # Find the last "User:" section and try to preserve it
        user_sections = prompt.split("User:")
        if len(user_sections) > 1:
            # Keep system prompt + last user section
            system_part = user_sections[0] + "User:"
            user_part = user_sections[-1]
            
            # If system part is too long, truncate it
            if len(system_part) > max_chars * 0.7:  # Reserve 30% for user part
                system_part = system_part[:int(max_chars * 0.7)]
            
            # Combine and truncate if still too long
            truncated = system_part + user_part
            if len(truncated) > max_chars:
                truncated = truncated[:max_chars]
            
            return truncated
        else:
            # No clear user section, truncate from end
            return prompt[:max_chars]
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat completion response using llama.cpp.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        start_time = time.time()
        
        try:
            # Check memory usage
            self._check_memory_usage()
            
            # Convert messages to prompt format
            formatted_prompt = self._convert_messages_to_prompt(messages)
            
            # Merge generation args with kwargs
            generation_kwargs = self.generation_args.copy()
            generation_kwargs.update(kwargs)
            
            # Get context length from device config
            device_config = self._get_device_specific_config()
            context_length = device_config.get("context_length", 4096)
            
            # Validate context window and adjust if needed
            max_new_tokens = generation_kwargs.get("max_new_tokens", 2048)
            validated_prompt, validated_max_tokens = self._validate_context_window(
                formatted_prompt, max_new_tokens, context_length
            )
            
            # Generate response using llama.cpp
            self.logger.debug(f"Generating chat completion for prompt: {validated_prompt[:100]}...")
            
            response = self.model(
                validated_prompt,
                max_tokens=validated_max_tokens,
                temperature=generation_kwargs.get("temperature", 0.7),
                top_p=generation_kwargs.get("top_p", 0.9),
                top_k=generation_kwargs.get("top_k", 40),
                repeat_penalty=generation_kwargs.get("repetition_penalty", 1.1),
                stop=["</s>", "<|endoftext|>", "User:", "System:"],  # Stop tokens
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
            
            self.logger.debug(f"Generated chat completion: {generated_text[:100]}...")
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}", exc_info=True)
            return ""
        finally:
            end_time = time.time()
            generation_time = end_time - start_time
            self.total_generation_time += generation_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the model interface."""
        return {
            "generation_count": self.generation_count,
            "total_tokens_generated": self.total_tokens_generated,
            "total_generation_time": self.total_generation_time,
            "average_tokens_per_generation": (
                self.total_tokens_generated / self.generation_count 
                if self.generation_count > 0 else 0
            ),
            "average_time_per_generation": (
                self.total_generation_time / self.generation_count 
                if self.generation_count > 0 else 0
            )
        }
