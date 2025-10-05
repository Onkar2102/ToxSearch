"""
mlm_operator.py

LLM-based Masked Language Model operator for text mutation.
Step 1: Randomly mask max_variants words with <MASKED_N>
Step 2: Ask LLM for each mask replacement one by one
Step 3: Apply replacements and return completed text

Author: Onkar Shelar (os9660@rit.edu)
"""

import random
from typing import List, Optional, Dict, Any, Tuple

try:
    from ea.VariationOperators import VariationOperator
except Exception:
    from VariationOperators import VariationOperator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class MLMOperator(VariationOperator):
    """
        Simplified LLM Mask-Fill Operator with 3-Step Process

    Design:
      - Step 1: Randomly mask m words with <MASKED_N>, where m = min(max_variants, word_count)
            - Step 2: Ask the LLM for each mask replacement individually (one-by-one prompts)
            - Step 3: Apply all replacements and return completed text
      - Returns exactly ONE completed variant (wrapped in a list to match interface)
    """

    def __init__(self, log_file: Optional[str] = None, max_variants: int = 3, seed: Optional[int] = 42):
        super().__init__("MLM", "mutation", "Enhanced 3-step mask and LLM fill with retry logic")
        self.logger = get_logger(self.name, log_file)
        
        # Improved parameter validation
        self.max_variants = self._validate_max_variants(max_variants)
        self.rng = random.Random(seed)
        
        # Initialize generator with better error handling
        try:
            from ea.operator_helpers import get_generator
            self.generator = get_generator()
            self.logger.info(f"{self.name}: LLM generator initialized successfully")
        except Exception as e:
            self.logger.warning(f"{self.name}: LLM generator unavailable: {e}")
            self.generator = None
        
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
        self.logger.debug(f"{self.name}: STEP 1 - Masked (m={m}) → {masked_text[:200]}{'...' if len(masked_text)>200 else ''}")
        self.logger.debug(f"{self.name}: Mask mapping: {mask_mapping}")
        
        # Store for debugging
        self._last_mask_mapping = mask_mapping
        self._last_masked_text = masked_text
        
        return masked_text, mask_mapping



    





    def _llm_fill_one_by_one(self, masked_text: str, mask_mapping: Dict[int, str]) -> str:
        """
        Step 2: Ask LLM for each mask replacement one by one, then apply manually.
        
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

        self.logger.debug(f"{self.name}: STEP 2 - Asking LLM for each mask replacement individually")
        
        # Store replacements as we get them
        replacements = {}
        all_responses = []
        
        # Ask for each mask replacement individually
        for mask_num, original_word in mask_mapping.items():
            mask_token = f"<MASKED_{mask_num}>"
            
            # Simple, clear prompt for one replacement
            prompt = f"""Replace {mask_token} with one word that fits the context.

Text: "{masked_text}"

Original word: "{original_word}"

Reply with just one replacement word:"""

            self.logger.debug(f"{self.name}: Asking for replacement of {mask_token} (original: '{original_word}')")
            
            try:
                response = self.generator.generate_response(prompt)
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
        
        self.logger.info(f"{self.name}: STEP 2 SUCCESS - Applied replacements: {replacements}")
        self._last_completed_text = completed_text
        return completed_text

    def apply(self, text: str) -> List[str]:
        """
        Apply the 3-step MLM process to generate text variants.
        
        Args:
            text: Input text to process
            
        Returns:
            List containing one completed variant
        """
        try:
            # Handle edge cases
            if not text or not text.strip():
                self.logger.debug(f"{self.name}: Empty input, returning as-is")
                return [text]
            
            self.logger.info(f"{self.name}: Starting 3-step MLM process for text: '{text[:50]}...'")
            
            # Step 1: Mask words with numbered tokens
            masked_text, mask_mapping = self._mask_once(text)
            
            if not mask_mapping:
                self.logger.info(f"{self.name}: No words masked, returning original text")
                return [text]
            
            # Step 2: Get replacements for each mask one by one
            completed_text = self._llm_fill_one_by_one(masked_text, mask_mapping)
            
            # Step 3: Validate and return result
            if completed_text != masked_text and "<MASKED_" not in completed_text:
                self.logger.info(f"{self.name}: STEP 3 SUCCESS - Generated variant: '{completed_text[:50]}...'")
                return [completed_text]
            else:
                self.logger.warning(f"{self.name}: STEP 3 FAILED - Returning original text")
                return [text]
                
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed: {e}")
            return [text]