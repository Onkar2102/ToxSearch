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
from .model_interface import LlamaCppChatInterface

# Get the functions at module level to avoid repeated calls
get_logger, _, _, _ = get_custom_logging()

class PromptGenerator:
    """
    Prompt generator using v1/chat/completions interface for prompt generation and modification.
    
    This class is specifically designed for operators and evolutionary algorithms
    to generate/modify prompts using task-specific templates with chat completions.
    """
    
    def __init__(self, model_key="prompt_generator", config_path="config/PGConfig.yaml", log_file: Optional[str] = None):
        self.log_file = log_file
        self.logger = get_logger("PromptGenerator", self.log_file)
        self.logger.debug(f"Logger correctly initialized with log_file: {self.log_file}")

        # Load PG config
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
        
        # Task-specific templates and unified LLM config from PG config
        self.task_templates = config.get("task_templates", {})
        self.llm_config = config.get("llm_config", {})
        self.operator_config = config.get("operator_config", {})

        # Performance tracking attributes
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0



    def generate_raw_response(self, prompt: str, generation_kwargs: Dict[str, Any] = None) -> str:
        """Generate raw model response without any template manipulation using chat completions."""
        try:
            # Create simple user message
            messages = [{"role": "user", "content": prompt}]
            
            # Generate with chat completions
            generated_text = self.model_interface.chat_completion(messages, **(generation_kwargs or {}))
            
            self.logger.debug(f"Raw response for '{prompt[:50]}...': '{generated_text[:100]}...'")
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Raw response generation failed: {e}")
            return f"Error: {e}"

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Direct access to chat completions for operators."""
        return self.model_interface.chat_completion(messages, **kwargs)



    
    def _extract_content_from_xml_tags(self, response: str, tag_name: str) -> str:
        """Extract content from XML tags with robust error handling and validation."""
        try:
            import re
            
            # Clean the response first
            response = response.strip()
            
            # Try exact tag matching first (case-sensitive)
            pattern = f'<{tag_name}>(.*?)</{tag_name}>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and self._validate_extracted_content(content, tag_name):
                    self.logger.debug(f"Successfully extracted {tag_name} content: {content[:50]}...")
                    return content
            
            # Try case-insensitive matching
            pattern = f'<{tag_name.lower()}>(.*?)</{tag_name.lower()}>'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and self._validate_extracted_content(content, tag_name):
                    self.logger.debug(f"Successfully extracted {tag_name} content (case-insensitive): {content[:50]}...")
                    return content
            
            # Try with whitespace tolerance
            pattern = f'<{tag_name}\\s*>(.*?)</{tag_name}\\s*>'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and self._validate_extracted_content(content, tag_name):
                    self.logger.debug(f"Successfully extracted {tag_name} content (whitespace-tolerant): {content[:50]}...")
                    return content
            
                # Try partial tag matching (in case of typos) - only for longer tag names
                if len(tag_name) > 3:
                    pattern = f'<{tag_name[:3]}.*?>(.*?)</{tag_name[:3]}.*?>'
                    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        if content and self._validate_extracted_content(content, tag_name):
                            self.logger.debug(f"Successfully extracted {tag_name} content (partial match): {content[:50]}...")
                            return content
                
                # Try to extract content from malformed XML (e.g., "opinions</synomics>")
                if tag_name in ['synonyms', 'antonyms']:
                    # Look for patterns like "word</synomics>" or "word</antonyms>"
                    pattern = f'([a-zA-Z]+)</{tag_name[:3]}.*?>'
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        if content and self._validate_extracted_content(content, tag_name):
                            self.logger.debug(f"Successfully extracted {tag_name} content (malformed XML): {content[:50]}...")
                            return content
            
            # Log the failed extraction for debugging
            self.logger.warning(f"Failed to extract valid {tag_name} content from response: {response[:200]}...")
            return ""
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {tag_name} tags: {e}")
            return ""
    
    def _validate_extracted_content(self, content: str, tag_name: str) -> bool:
        """Validate extracted content based on tag type."""
        if not content or len(content.strip()) < 2:
            return False
            
        # For question-related tags, ensure it's a question
        if tag_name in ['variant', 'paraphrase', 'modified', 'trans', 'triple_bracket']:
            # Must be a question - accept various question marks from different languages
            # English: ?, Japanese/Chinese: ？, Arabic: ؟, Greek: ;
            question_marks = ['?', '？', '؟', ';']
            if not any(content.endswith(qm) for qm in question_marks):
                self.logger.warning(f"Extracted content does not end with question mark: {content}")
                return False
                
        # # For word lists (synonyms, antonyms), validate single word format
        # elif tag_name in ['synonyms', 'antonyms']:
        #     # Should be a single word, not JSON array
        #     if len(content.split()) > 1:
        #         self.logger.warning(f"{tag_name} should be single word: {content}")
        #         return False
        #     if not content.isalpha() or len(content.strip()) < 2:
        #         self.logger.warning(f"Invalid {tag_name} word: {content}")
        #         return False
                
        # # For single word replacements
        # elif tag_name == 'replacement':
        #     # Allow multi-word replacements for MLM operator
        #     if len(content.split()) > 3:
        #         self.logger.warning(f"Replacement too long: {content}")
        #         return False
                
        return True

    def _extract_content_from_triple_brackets(self, response: str) -> str:
        """Extract content from triple brackets (<<< ... >>>) with robust error handling and validation."""
        try:
            import re
            
            # Clean the response first
            response = response.strip()
            
            # Try exact triple bracket matching
            pattern = r'<<<(.*?)>>>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and self._validate_extracted_content(content, "triple_bracket"):
                    self.logger.debug(f"Successfully extracted triple bracket content: {content[:50]}...")
                    return content
            
            # Try with whitespace tolerance around brackets
            pattern = r'<<<\s*(.*?)\s*>>>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and self._validate_extracted_content(content, "triple_bracket"):
                    self.logger.debug(f"Successfully extracted triple bracket content (whitespace-tolerant): {content[:50]}...")
                    return content
            
            # Try case-insensitive matching (in case of typos)
            pattern = r'<<<(.*?)>>>'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and self._validate_extracted_content(content, "triple_bracket"):
                    self.logger.debug(f"Successfully extracted triple bracket content (case-insensitive): {content[:50]}...")
                    return content
            
            # Log the failed extraction for debugging
            self.logger.warning(f"Failed to extract valid triple bracket content from response: {response[:200]}...")
            return ""
            
        except Exception as e:
            self.logger.error(f"Error extracting content from triple brackets: {e}")
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
