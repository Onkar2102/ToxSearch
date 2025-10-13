"""
PromptGenerator.py

A specialized text generator for prompt generation using task-specific templates.
This class is used by all operators and evolutionary algorithms to generate/modify prompts.
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

class PromptGenerator:
    """
    Prompt generator using llama.cpp for prompt generation and modification.
    
    This class is specifically designed for operators and evolutionary algorithms
    to generate/modify prompts using task-specific templates.
    """
    
    _MODEL_CACHE = {}
    
    def __init__(self, model_key="prompt_generator", config_path="config/PGConfig.yaml", log_file: Optional[str] = None):
        self.log_file = log_file
        self.logger = get_logger("PromptGenerator", self.log_file)
        self.logger.debug(f"Logger correctly initialized with log_file: {self.log_file}")

        # Load PG config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if model_key not in config:
            raise ValueError(f"Model '{model_key}' not found in configuration.")
        self.model_cfg = config[model_key]

        # Memory management settings
        self.enable_memory_cleanup = self.model_cfg.get("enable_memory_cleanup", True)
        self.max_memory_usage_gb = self.model_cfg.get("max_memory_usage_gb", 6.0)

        # Performance tracking attributes
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        
        # Load model
        model_path = self.model_cfg["name"]
        if model_path not in self._MODEL_CACHE:
            self.logger.info(f"Loading prompt generation model: {model_path}")
            self._load_model(model_path)
        else:
            self.logger.info(f"Using cached prompt generation model: {model_path}")

        self.model = self._MODEL_CACHE[model_path]
        self.generation_args = self.model_cfg.get("generation_args", {})
        
        # Task-specific templates and unified LLM config from PG config
        self.task_templates = config.get("task_templates", {})
        self.llm_config = config.get("llm_config", {})
        self.operator_config = config.get("operator_config", {})

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
            self.logger.info("Loading prompt generation model with llama.cpp...")
            model = Llama(**llama_params)
            
            self._MODEL_CACHE[model_path] = model
            self.logger.info(f"Prompt generation model loaded successfully: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load prompt generation model: {e}")
            # Create mock model as fallback
            self.logger.info("Creating mock model as fallback")
            mock_model = self._create_mock_model()
            self._MODEL_CACHE[model_path] = mock_model

    def _create_mock_model(self):
        """Create a mock model for testing when real model is not available."""
        class MockModel:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": f"Mock response for: {prompt[:50]}..."}]}
        return MockModel()

    def _check_memory_usage(self):
        """Check memory usage and cleanup if necessary."""
        try:
            current_time = time.time()
            if current_time - self.last_memory_check < self.memory_check_interval:
                return
            
            memory_info = psutil.virtual_memory()
            memory_usage_gb = memory_info.used / (1024**3)
            
            if memory_usage_gb > self.max_memory_usage_gb and self.enable_memory_cleanup:
                self.logger.warning(f"Memory usage ({memory_usage_gb:.1f}GB) exceeds limit ({self.max_memory_usage_gb}GB)")
                self._cleanup_memory()
            
            self.last_memory_check = current_time
        except Exception as e:
            self.logger.warning(f"Memory check failed: {e}")

    def _cleanup_memory(self):
        """Clean up memory by forcing garbage collection."""
        try:
            import gc
            gc.collect()
            self.logger.info("Memory cleanup completed")
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

    def generate_prompt(self, prompt: str, task_type: str, **kwargs) -> str:
        """Generate a prompt variant using task-specific templates."""
        start_time = time.time()
        
        try:
            # Get task template from PG config
            template = self.task_templates.get(task_type, "")
            if not template:
                self.logger.warning(f"No template found for task type: {task_type}")
                return ""
            
            # Handle nested templates (like translation)
            if isinstance(template, dict):
                # For nested templates, use the first available template
                template = next(iter(template.values())) if template else ""
            
            # Format prompt with template and additional kwargs
            try:
                # Try to format with prompt and any additional kwargs
                format_kwargs = {"prompt": prompt}
                format_kwargs.update(kwargs)
                formatted_prompt = template.format(**format_kwargs)
            except KeyError as e:
                # Template doesn't have the required placeholder, use prompt as-is
                self.logger.debug(f"Template missing placeholder {e}, using prompt as-is")
                formatted_prompt = prompt
            
            # Get generation args from unified LLM config
            generation_kwargs = {
                **self.generation_args,
                **self.llm_config,  # Use unified LLM config for all operators
            }
            
            # Generate response using llama.cpp
            self.logger.debug(f"Generating prompt variant for task: {task_type}")
            
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
            
            self.logger.debug(f"Generated prompt variant: {generated_text[:100]}...")
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Prompt generation failed: {e}", exc_info=True)
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

    def generate_response(self, prompt: str, task_type: str, **kwargs) -> str:
        """Generate a response using task-specific templates (for operators)."""
        return self.generate_prompt(prompt, task_type, **kwargs)

    def generate_raw_response(self, prompt: str, generation_kwargs: Dict[str, Any] = None) -> str:
        """Generate raw model response without any template manipulation."""
        try:
            if generation_kwargs is None:
                generation_kwargs = {
                    **self.generation_args,
                    **self.llm_config,
                }
            
            # Generate with custom args
            response = self.model(
                prompt,
                max_tokens=generation_kwargs.get("max_new_tokens", 2048),
                temperature=generation_kwargs.get("temperature", 0.7),
                top_p=generation_kwargs.get("top_p", 0.9),
                top_k=generation_kwargs.get("top_k", 40),
                repeat_penalty=generation_kwargs.get("repetition_penalty", 1.1),
                stop=["</s>", "<|endoftext|>"],
                echo=False,
            )
            
            # Extract text from response
            if isinstance(response, dict) and 'choices' in response:
                generated_text = response['choices'][0]['text']
            elif isinstance(response, str):
                generated_text = response
            else:
                generated_text = str(response)
            
            generated_text = generated_text.strip()
            self.logger.debug(f"Raw response for '{prompt[:50]}...': '{generated_text[:100]}...'")
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Raw response generation failed: {e}")
            return f"Error: {e}"

    def translate(self, text: str, target_language: str, source_language: str = "English") -> str:
        """High-precision translation using task templates."""
        try:
            # Get translation template
            templates = self.task_templates.get("translation", {})
            if str(source_language).strip().lower() == "english":
                template = templates.get("en_to_target")
            else:
                template = templates.get("target_to_en")

            if not template:
                self.logger.warning("Translation template missing; returning original text.")
                return text

            prompt = template.format(
                text=text,
                target_language=target_language,
                source_language=source_language
            )

            # Generate response
            response = self.generate_raw_response(prompt)

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
        """High-precision paraphrasing using task templates."""
        try:
            # Get paraphrasing template
            template = self.task_templates.get("paraphrasing", "")

            if not template:
                self.logger.warning("Paraphrasing template missing; returning original text.")
                return text

            prompt = template.format(
                north_star_metric=north_star_metric,
                original_prompt=text,
                generated_output=generated_output,
                current_score=current_score
            )

            # Generate response
            response = self.generate_raw_response(prompt)

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
        """High-precision stylistic mutation using task templates."""
        try:
            # Get stylistic mutation template
            template = self.task_templates.get("stylistic_mutation", "")

            if not template:
                self.logger.warning("Stylistic mutation template missing; returning original text.")
                return text

            prompt = template.format(
                original_text=text,
                style_attribute=style_attribute
            )

            # Generate response
            response = self.generate_raw_response(prompt)

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

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the prompt generator."""
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
