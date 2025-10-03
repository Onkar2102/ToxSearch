"""
bert_mlm_operator.py

Author: Onkar Shelar os9660@rit.edu

This module contains the BERT MLM (Masked Language Model) operator for the evolutionary algorithm.
This mutation operator performs word replacement using BERT's masked language model predictions.

Author: EOST CAM LLM Team
Version: 1.0
"""

import torch
from typing import List, Optional, Dict, Any, Tuple
from transformers import BertTokenizer, BertForMaskedLM

try:
    from ea.VariationOperators import VariationOperator
except Exception:
    from VariationOperators import VariationOperator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class BertMLMOperator(VariationOperator):
    """
    BERT-based Masked Language Model operator for word replacement.
    
    This mutation operator uses BERT's masked language model to replace words
    in the input text. It masks each word position and uses BERT to predict
    the most likely replacements based on context.
    
    Attributes:
        name (str): Operator name "BertMLM"
        operator_type (str): "mutation" (single parent operator)
        description (str): Description of the operator's functionality
        logger: Logger instance for debugging and monitoring
        tokenizer: BERT tokenizer instance
        model: BERT masked language model instance
        
    Methods:
        apply(text): Generate variants by replacing words with BERT predictions
        
    Note:
        Each instance loads its own BERT model, which may impact memory usage.
        Consider implementing model caching for better efficiency.
        
    Example:
        >>> operator = BertMLMOperator()
        >>> variants = operator.apply("Write a story about a brave knight")
        >>> print(variants)
        ['Write a story about a medieval knight', 'Write a story about a brave warrior']
    """
    
    def __init__(self, log_file=None):
        """
        Initialize the BERT MLM operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Loads BERT model and tokenizer from Hugging Face transformers.
        """
        super().__init__("BertMLM", "mutation", "Uses BERT MLM to replace one word.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    def apply(self, text: str) -> List[str]:
        """
        Generate variants by replacing words with BERT MLM predictions.
        
        This method:
        1. Splits the input text into words
        2. For each word position, masks the word with [MASK] token
        3. Uses BERT to predict the most likely replacements
        4. Creates variants by replacing original words with predictions
        5. Returns up to 3 variants (limited by limit_variants function)
        
        Args:
            text (str): Input text to generate variants from
            
        Returns:
            List[str]: List of variant texts (maximum 3)
            
        Example:
            >>> operator = BertMLMOperator()
            >>> variants = operator.apply("Write a story about a brave knight")
            >>> print(variants)
            ['Write a story about a medieval knight', 'Write a story about a brave warrior']
        """
        try:
            words = text.split()
            if not words:
                return [text]

            variants = set()
            for idx in range(len(words)):
                original = words[idx]
                masked_words = words.copy()
                masked_words[idx] = "[MASK]"
                masked_text = " ".join(masked_words)

                inputs = self.tokenizer(masked_text, return_tensors="pt")
                with torch.no_grad():
                    logits = self.model(**inputs).logits

                mask_idx = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                topk = torch.topk(logits[0, mask_idx], k=5, dim=-1).indices[0].tolist()

                for token_id in topk:
                    new_word = self.tokenizer.decode([token_id]).strip()
                    mutated = words.copy()
                    mutated[idx] = new_word
                    result = " ".join(mutated).strip()
                    if result.lower() != text.strip().lower():
                        variants.add(result)

            result_variants = list(variants) if variants else [text]
            
            # Limit variants to maximum of 3
            from .operator_helpers import limit_variants
            limited_variants = limit_variants(result_variants, max_variants=3)
            self.logger.debug(f"{self.name}: Generated {len(result_variants)} variants, limited to {len(limited_variants)} via BERT MLM from: '{text[:60]}...'")
            return limited_variants
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed with error: {e}")
            return [text]
