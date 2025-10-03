"""
pos_aware_synonym_replacement.py


This module contains the POS-aware synonym replacement operator for the evolutionary algorithm.
This mutation operator performs BERT-based synonym replacement using POS (Part-of-Speech) tagging.
"""

import random
import torch
import spacy
from nltk.corpus import wordnet as wn
from typing import List, Optional, Dict, Any, Tuple
from transformers import BertTokenizer, BertForMaskedLM

try:
    from ea.VariationOperators import VariationOperator
except Exception:
    from VariationOperators import VariationOperator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()

nlp = spacy.load("en_core_web_sm")


class POSAwareSynonymReplacement(VariationOperator):
    """
    POS-aware synonym replacement operator using spaCy POS tagging and BERT MLM.
    
    This mutation operator identifies parts of speech in the input text and replaces
    them with semantically appropriate synonyms using BERT's masked language model.
    
    Attributes:
        name (str): Operator name "POSAwareSynonymReplacement"
        operator_type (str): "mutation" (single parent operator)
        description (str): Description of the operator's functionality
        logger: Logger instance for debugging and monitoring
        tokenizer: BERT tokenizer instance
        model: BERT masked language model instance
        
    Methods:
        apply(text): Generate variants by replacing POS-tagged words with synonyms
        
    Example:
        >>> operator = POSAwareSynonymReplacement()
        >>> variants = operator.apply("Write a story about a brave knight")
        >>> print(variants)
        ['Write a story about a courageous warrior', 'Write a story about a heroic hero']
    """
    
    def __init__(self, log_file=None):
        """
        Initialize the POS-aware synonym replacement operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Loads BERT model and tokenizer from Hugging Face transformers.
        """
        super().__init__("POSAwareSynonymReplacement", "mutation", "BERT-based synonym replacement based on spaCy POS.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
        # BERT tokenizer/model for MLM
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    def apply(self, text: str) -> List[str]:
        """
        Generate variants by replacing words based on POS tagging with BERT synonyms.
        
        This method:
        1. Parses text using spaCy for POS tagging
        2. Identifies target parts of speech (ADJ, VERB, NOUN, ADV, etc.)
        3. For each target POS word, masks it and uses BERT to predict alternatives
        4. Creates variants by replacing original words with top predictions
        5. Returns up to 3 variants (limited by limit_variants function)
        
        Args:
            text (str): Input text to generate variants from
            
        Returns:
            List[str]: List of variant texts (maximum 3)
            
        Example:
            >>> operator = POSAwareSynonymReplacement()
            >>> variants = operator.apply("Write a story about a brave knight")
            >>> print(variants)
            ['Write a story about a courageous warrior', 'Write a story about a heroic hero']
        """
        try:
            doc = nlp(text)
            words = [t.text for t in doc]
            variants = set()

            pos_map = {
                "ADJ": wn.ADJ,
                "VERB": wn.VERB,
                "NOUN": wn.NOUN,
                "ADV": wn.ADV,
                "ADP": wn.ADJ,
                "INTJ": wn.ADJ
            }
            target_pos = set(pos_map.keys())
            pos_counts = {pos: 0 for pos in target_pos}
            replacement_log = []

            for i, token in enumerate(doc):
                if token.pos_ not in target_pos:
                    continue
                pos_counts[token.pos_] += 1

                masked_words = words.copy()
                masked_words[i] = "[MASK]"
                masked_text = " ".join(masked_words)
                inputs = self.tokenizer(masked_text, return_tensors="pt")
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                mask_idx = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                topk = torch.topk(logits[0, mask_idx], k=10, dim=-1).indices[0].tolist()

                for token_id in topk:
                    new_word = self.tokenizer.decode([token_id]).strip()
                    self.logger.debug(f"{self.name}: Attempting replacement for '{token.text}' (POS: {token.pos_}) with '{new_word}'")
                    if new_word.lower() != token.text.lower():
                        mutated = words.copy()
                        mutated[i] = new_word
                        variant = " ".join(mutated)
                        if variant.lower().strip() != text.lower().strip():
                            variants.add(variant)
                            replacement_log.append((token.text, new_word, token.pos_))

            result_variants = list(variants) if variants else [text]
            for pos, count in pos_counts.items():
                self.logger.debug(f"{self.name}: Found {count} tokens with POS {pos}")
            for original, new, pos in replacement_log:
                self.logger.debug(f"{self.name}: Replaced '{original}' with '{new}' (POS: {pos})")
            
            # Limit variants to maximum of 3
            from .operator_helpers import limit_variants
            limited_variants = limit_variants(result_variants, max_variants=3)
            self.logger.debug(f"{self.name}: Generated {len(result_variants)} variants, limited to {len(limited_variants)} using BERT synonym substitution for POS-aware replacement from: '{text[:60]}...'")
            return limited_variants
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed with error: {e}")
            return [text]
