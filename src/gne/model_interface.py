"""
Model Interface Abstraction Layer

Provides OpenAI-compatible v1/chat/completions interface for model-agnostic architecture.
Currently implements llama_cpp provider with chat completions support.
"""

import os
import time
import psutil
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
        
        # Use absolute path for caching
        if absolute_model_path not in self._MODEL_CACHE:
            self.logger.info(f"Loading llama.cpp model: {absolute_model_path}")
            self._load_model(absolute_model_path)
            # Track access count for new model
            self._MODEL_CACHE_ACCESS_COUNT[absolute_model_path] = 1
        else:
            self.logger.info(f"Using cached llama.cpp model: {absolute_model_path}")
            # Increment access count
            self._MODEL_CACHE_ACCESS_COUNT[absolute_model_path] = self._MODEL_CACHE_ACCESS_COUNT.get(absolute_model_path, 0) + 1
        
        # Cleanup cache if too many models
        self._cleanup_model_cache_if_needed()
        
        self.model = self._MODEL_CACHE[absolute_model_path]
        self.generation_args = model_cfg.get("generation_args", {})
        
        # Memory monitoring
        self.last_memory_check = time.time()
        self.memory_check_interval = 60  # Check memory every 60 seconds
    
    def _cleanup_model_cache_if_needed(self):
        """Clean up unused models from cache"""
        if len(self._MODEL_CACHE) > 2:  # Keep only 2 models max
            # Remove least recently used model
            least_used = min(self._MODEL_CACHE_ACCESS_COUNT.items(), key=lambda x: x[1])
            del self._MODEL_CACHE[least_used[0]]
            del self._MODEL_CACHE_ACCESS_COUNT[least_used[0]]
            self.logger.info(f"Removed model {least_used[0]} from cache (access count: {least_used[1]})")
    
    def _load_model(self, model_path: str):
        """Load model using llama.cpp with optimizations."""
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
            
            # Configure llama.cpp parameters
            llama_params = {
                "model_path": model_path,
                "n_ctx": 4096,  # Context window
                "n_threads": None,  # Auto-detect
                "n_gpu_layers": -1,  # Use all available GPU layers (MPS on Apple Silicon)
                "verbose": False,
                "use_mmap": True,  # Memory mapping for efficiency
                "use_mlock": False,  # Don't lock memory
                "low_vram": False,  # Not needed for MPS
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
            
            # Generate response using llama.cpp
            self.logger.debug(f"Generating chat completion for prompt: {formatted_prompt[:100]}...")
            
            response = self.model(
                formatted_prompt,
                max_tokens=generation_kwargs.get("max_new_tokens", 2048),
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
