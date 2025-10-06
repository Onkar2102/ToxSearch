"""
mlm_operator.py

LLM-based Masked Language Model operator for text mutation.

This operator implements a mask-and-fill approach using a local LLaMA model.
It randomly masks words in the input text and uses the LLM to generate
appropriate replacements, creating contextually coherent text variants.

Process:
1. Randomly mask up to max_variants words with placeholder tokens
2. Generate replacement suggestions using LLaMA for each masked position
3. Apply replacements and return the completed text variant

Author: Onkar Shelar (os9660@rit.edu)
"""

import random
from typing import List, Optional, Dict, Any, Tuple

from .VariationOperators import VariationOperator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class MLMOperator(VariationOperator):
    """
    LLM-based masked language model operator for text mutation.
    
    This operator randomly masks words in the input text and uses a local
    LLaMA model to generate contextually appropriate replacements. The
    masking and replacement process creates coherent text variants while
    preserving overall meaning and structure.
    
    Process:
    1. Randomly mask up to max_variants words with numbered placeholders
    2. Use LLaMA to generate replacement words for each masked position
    3. Apply all replacements to create the final text variant
    
    Returns:
        List[str]: Single completed text variant (wrapped in list for interface compatibility)
    
    Attributes:
        max_variants (int): Maximum number of words to mask per operation
        rng: Random number generator for reproducible word selection
        generator: Local LLaMA model instance for replacement generation
    """

    def __init__(self, log_file: Optional[str] = None, max_variants: int = 3, seed: Optional[int] = 42):
        super().__init__("MLM", "mutation", "LLM-based masked language model operator for contextual word replacement")
        self.logger = get_logger(self.name, log_file)
        
        # Improved parameter validation
        self.max_variants = self._validate_max_variants(max_variants)
        self.rng = random.Random(seed)
        
        # Initialize generator
        from .operator_helpers import get_generator
        self.generator = get_generator()
        self.logger.info(f"{self.name}: LLM generator initialized successfully")
        
        # Debug/trace attributes for tests and observability
        self._last_mask_mapping: Dict[int, str] = {}
        self._last_masked_text: str = ""
        self._last_structured_prompt: str = ""
        self._last_raw_response: str = ""
        self._last_parsed_result: Optional[Dict[str, Any]] = None
        self._last_completed_text: str = ""
        
        self.logger.info(f"{self.name}: Initialized with max_variants={self.max_variants}, seed={seed}")

    def _validate_max_variants(self, max_variants: Any) -> int:
        """Validate and convert max_variants to positive integer."""
        try:
            if isinstance(max_variants, str):
                max_variants = int(max_variants)
            elif isinstance(max_variants, float):
                max_variants = int(max_variants)
            
            val = max(1, int(max_variants))
            if val < 1:
                self.logger.warning(f"{self.name}: max_variants < 1, setting to 1")
                return 1
            return val
        except (ValueError, TypeError) as e:
            self.logger.warning(f"{self.name}: Invalid max_variants '{max_variants}', using default 3: {e}")
            return 3

    def _mask_once(self, text: str) -> Tuple[str, Dict[int, str]]:
        """
        Step 1: Mask words with numbered tokens and track original words.
        
        Args:
            text: Input text to mask
            
        Returns:
            Tuple of (masked_text, mask_mapping) where mask_mapping maps mask_number -> original_word
        """
        words = text.split()
        if not words:
            return text, {}
        
        m = min(self.max_variants, len(words))
        try:
            idxs = set(self.rng.sample(range(len(words)), m))
        except ValueError:
            idxs = set()
        
        # Create numbered masks and track original words
        mask_mapping = {}
        mask_counter = 1
        masked_words = []
        
        for i, word in enumerate(words):
            if i in idxs:
                mask_token = f"<MASKED_{mask_counter}>"
                mask_mapping[mask_counter] = word
                masked_words.append(mask_token)
                mask_counter += 1
            else:
                masked_words.append(word)
        
        masked_text = " ".join(masked_words)
        self.logger.debug(f"{self.name}: Masked {m} words → {masked_text[:200]}{'...' if len(masked_text)>200 else ''}")
        self.logger.debug(f"{self.name}: Mask mapping: {mask_mapping}")
        
        # Store for debugging
        self._last_mask_mapping = mask_mapping
        self._last_masked_text = masked_text
        
        return masked_text, mask_mapping



    





    def _get_llm_replacements_sequentially(self, masked_text: str, mask_mapping: Dict[int, str]) -> str:
        """
        Generate LLM replacements for each masked token individually.
        
        Args:
            masked_text: Text with numbered mask tokens
            mask_mapping: Mapping of mask numbers to original words
            
        Returns:
            Completed text with masks replaced, or original masked text if failed
        """
        if not self.generator:
            self.logger.warning(f"{self.name}: No generator; returning masked text")
            return masked_text

        if not mask_mapping:
            self.logger.debug(f"{self.name}: No masks to replace")
            return masked_text

        self.logger.debug(f"{self.name}: Asking LLM for each mask replacement individually")
        
        # Store replacements as we get them
        replacements = {}
        all_responses = []
        
        # Ask for each mask replacement individually
        for mask_num, original_word in mask_mapping.items():
            mask_token = f"<MASKED_{mask_num}>"
            
            # Get prompt template from config
            template = self.generator.task_templates.get("mlm_mask_filling", {}).get("single_word_replacement")
            if template:
                prompt = template.format(
                    mask_token=mask_token,
                    masked_text=masked_text,
                    original_word=original_word
                )
            else:
                # Fallback to original prompt if template not found
                prompt = f"""Replace {mask_token} with one word that fits the context.

Text: "{masked_text}"

Original word: "{original_word}"

Reply with just one replacement word:"""

            self.logger.debug(f"{self.name}: Asking for replacement of {mask_token} (original: '{original_word}')")
            
            try:
                response = self.generator.generate_response(prompt, task_type="mutation_crossover")
                all_responses.append(f"{mask_token}: {response}")
                
                if response:
                    # Clean up the response to get just the word
                    replacement = response.strip().strip('"').strip("'").split()[0]  # Take first word only
                    
                    # Basic validation: should be a single word, no special tokens
                    if replacement and len(replacement.split()) == 1 and "<MASKED_" not in replacement:
                        replacements[mask_num] = replacement
                        self.logger.debug(f"{self.name}: {mask_token} -> '{replacement}'")
                    else:
                        self.logger.warning(f"{self.name}: Invalid replacement for {mask_token}: '{replacement}', using original")
                        replacements[mask_num] = original_word
                else:
                    self.logger.warning(f"{self.name}: Empty response for {mask_token}, using original")
                    replacements[mask_num] = original_word
                    
            except Exception as e:
                self.logger.error(f"{self.name}: Failed to get replacement for {mask_token}: {e}")  
                replacements[mask_num] = original_word
        
        # Store debug info
        self._last_structured_prompt = f"One-by-one prompts for {len(mask_mapping)} masks"
        self._last_raw_response = " | ".join(all_responses)
        self._last_parsed_result = {"replacements": {str(k): v for k, v in replacements.items()}}
        
        # Now manually apply all replacements
        completed_text = masked_text
        for mask_num, replacement in replacements.items():
            mask_token = f"<MASKED_{mask_num}>"
            completed_text = completed_text.replace(mask_token, replacement, 1)
        
        self.logger.info(f"{self.name}: Applied replacements: {replacements}")
        self._last_completed_text = completed_text
        return completed_text

    def apply(self, operator_input: Dict[str, Any]) -> List[str]:
        """
        Apply the 3-step MLM process to generate text variants.
        
        This method:
        1. Validates input format and extracts parent data
        2. Applies 3-step MLM process (mask, fill, complete)
        3. Returns MLM-generated variant if different from original
        4. Falls back to original text if MLM fails
        
        Args:
            operator_input (Dict[str, Any]): Operator input containing:
                - 'parent_data': Enriched parent genome dictionary containing:
                    - 'prompt': Original prompt text to process with MLM
                    - 'generated_text': Generated output from the prompt (optional)
                    - 'scores': Moderation scores dictionary
                    - 'north_star_score': Primary optimization metric score
                - 'max_variants': Maximum number of variants to generate
                
        Returns:
            List[str]: List containing MLM-generated prompt variants (or original if failed)
        """
        try:
            # Validate input format
            if not isinstance(operator_input, dict):
                self.logger.error(f"{self.name}: Input must be a dictionary")
                return []
            
            # Extract parent data
            parent_data = operator_input.get("parent_data", {})
            
            if not isinstance(parent_data, dict):
                self.logger.error(f"{self.name}: parent_data must be a dictionary")
                return []
            
            # Extract prompt from parent data
            text = parent_data.get("prompt", "")
            
            # Handle edge cases
            if not text or not text.strip():
                self.logger.debug(f"{self.name}: Empty input, returning as-is")
                return [text]
            
            self.logger.info(f"{self.name}: Starting 3-step MLM process for text: '{text[:50]}...'")
            
            # Mask words with numbered tokens
            masked_text, mask_mapping = self._mask_once(text)
            
            if not mask_mapping:
                self.logger.info(f"{self.name}: No words masked, returning original text")
                return [text]
            
            # Get LLM replacements for each mask
            completed_text = self._get_llm_replacements_sequentially(masked_text, mask_mapping)
            
            # Validate and return result
            if completed_text != masked_text and "<MASKED_" not in completed_text:
                self.logger.info(f"{self.name}: Generated variant: '{completed_text[:50]}...'")
                return [completed_text]
            else:
                self.logger.warning(f"{self.name}: Generation failed - Returning original text")
                return [text]
                
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed: {e}")
            return [text]