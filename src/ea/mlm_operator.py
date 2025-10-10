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

"""

import random
from typing import List, Optional, Dict, Any, Tuple

from .VariationOperators import VariationOperator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class MLMOperator(VariationOperator):
    """LLM-based masked language model operator for text mutation."""

    def __init__(self, log_file: Optional[str] = None, max_variants: int = 3, seed: Optional[int] = 42, generator=None):
        super().__init__("MLM", "mutation", "LLM-based masked language model operator for contextual word replacement")
        self.logger = get_logger(self.name, log_file)
        self.max_variants = self._validate_max_variants(max_variants)
        self.rng = random.Random(seed)
        self.generator = generator

    def _validate_max_variants(self, max_variants: Any) -> int:
        """Validate and convert max_variants to positive integer."""
        try:
            val = max(1, int(max_variants))
            return val
        except (ValueError, TypeError):
            self.logger.warning(f"{self.name}: Invalid max_variants '{max_variants}', using default 3")
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
            template = self.generator.task_templates.get("single_word_replacement", "")
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
                response = self.generator.generate_response(prompt, task_type="single_word_replacement")
                all_responses.append(f"{mask_token}: {response}")
                
                if response:
                    # Extract replacement from structured tags
                    import re
                    replacement_match = re.search(r'<replacement>(.*?)</replacement>', response, re.DOTALL)
                    if replacement_match:
                        replacement = replacement_match.group(1).strip()
                    else:
                        # Fallback: Extract word from response
                        replacement = self._extract_word_from_response(response, original_word)
                    
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
    
    def _extract_word_from_response(self, response: str, fallback_word: str) -> str:
        """Extract a single word from LLM response as fallback parsing."""
        try:
            import re
            # Look for quoted words first
            quoted_words = re.findall(r'"([^"]+)"', response)
            if quoted_words:
                for word in quoted_words:
                    stripped_word = word.strip()
                    if stripped_word and len(stripped_word.split()) == 1 and stripped_word.isalpha():
                        return stripped_word
            
            # Look for single words in the response
            words = re.findall(r'\b[a-zA-Z]{2,}\b', response)
            if words:
                for word in words:
                    if len(word) > 1 and word.isalpha():
                        return word
            
            return fallback_word
            
        except Exception as e:
            self.logger.debug(f"{self.name}: Failed to extract word from response: {e}")
            return fallback_word

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
                self.logger.debug(f"{self.name}: Generated variant: '{completed_text[:50]}...'")
                return [completed_text]
            else:
                self.logger.warning(f"{self.name}: Generation failed - Returning original text")
                return [text]
                
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed: {e}")
            return [text]