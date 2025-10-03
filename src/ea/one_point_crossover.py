"""
one_point_crossover.py

Author: Onkar Shelar os9660@rit.edu

This module contains the one-point crossover operator for the evolutionary algorithm.
This crossover operator swaps matching-position sentences between two parent prompts.

Author: EOST CAM LLM Team
Version: 1.0
"""

import nltk
from nltk.tokenize import sent_tokenize
from typing import List
import traceback

try:
    from ea.VariationOperators import VariationOperator
except Exception:
    from VariationOperators import VariationOperator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class OnePointCrossover(VariationOperator):
    """
    One-point crossover operator for prompt sentence swapping.
    
    This crossover operator performs sentence-based crossover between two parent prompts.
    It swaps sequences of sentences at matching positions to create new variants.
    
    Attributes:
        name (str): Operator name "OnePointCrossover"
        operator_type (str): "crossover" (multiple parents required)
        description (str): Description of the operator's functionality
        logger: Logger instance for debugging and monitoring
        
    Methods:
        apply(parent_texts): Generate crossover variants from parent texts
        
    Example:
        >>> operator = OnePointCrossover()
        >>> parents = ["First sentence. Second sentence.", "Third sentence. Fourth sentence."]
        >>> variants = operator.apply(parents)
        >>> print(variants)
        ['First sentence. Fourth sentence.', 'Third sentence. Second sentence.']
    """
    
    def __init__(self, log_file=None):
        """
        Initialize the one-point crossover operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Uses NLTK sentence tokenization for text analysis.
        """
        super().__init__("OnePointCrossover", "crossover", "Swaps matching-position sentences between two parents.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")

    def apply(self, parent_texts: List[str]) -> List[str]:
        """
        Generate crossover variants by swapping sentence sequences between parents.
        
        This method:
        1. Validates input format and parent count
        2. Tokenizes parent texts into sentences using NLTK
        3. Finds sentence sequence swap opportunities (1, 2, or 3 sentence swaps)
        4. Performs swaps at all possible positions
        5. Returns up to 3 variants (limited by limit_variants function)
        
        Args:
            parent_texts (List[str]): List of parent prompt texts (minimum 2 required)
            
        Returns:
            List[str]: List of crossover variant texts (maximum 3)
            
        Raises:
            Warning: If insufficient parents provided, logs warning and returns single parent
            
        Example:
            >>> operator = OnePointCrossover()
            >>> parents = ["First sentence. Second sentence.", "Third sentence. Fourth sentence."]
            >>> variants = operator.apply(parents)
            >>> print(variants)
            ['First sentence. Fourth sentence.', 'Third sentence. Second sentence.']
        """
        try:
            if not isinstance(parent_texts, list) or len(parent_texts) < 2:
                self.logger.warning(f"{self.name}: Insufficient parents for crossover.")
                return [parent_texts[0]] if parent_texts else []

            # Download required NLTK data if not present
            try:
                nltk.data.find('tokenizers/punkt')
            except FileNotFoundError:
                self.logger.info(f"{self.name}: Downloading NLTK punkt tokenizer...")
                nltk.download('punkt')
                nltk.download('punkt_tab')

            parent1_sentences = sent_tokenize(parent_texts[0])
            parent2_sentences = sent_tokenize(parent_texts[1])

            min_len = min(len(parent1_sentences), len(parent2_sentences))
            if min_len < 2:
                self.logger.warning(f"{self.name}: One or both parents have fewer than 2 sentences. Skipping.")
                return [parent_texts[0], parent_texts[1]]

            swap_counts = []
            if min_len >= 2:
                swap_counts.append(1)
            if min_len >= 3:
                swap_counts.append(2)
            if min_len >= 4:
                swap_counts.append(3)

            children = []

            for n in swap_counts:
                for start_idx in range(min_len - n + 1):
                    p1_variant = parent1_sentences.copy()
                    p2_variant = parent2_sentences.copy()

                    # Swap n sentences starting at position `start_idx`
                    p1_variant[start_idx:start_idx+n], p2_variant[start_idx:start_idx+n] = \
                        parent2_sentences[start_idx:start_idx+n], parent1_sentences[start_idx:start_idx+n]

                    child1 = " ".join(p1_variant).strip()
                    child2 = " ".join(p2_variant).strip()

                    children.append(child1)
                    children.append(child2)
                    self.logger.debug(f"{self.name}: Swapped {n} sentence(s) from position {start_idx} to create two variants.")

            # Limit variants to maximum of 3
            from .operator_helpers import limit_variants
            limited_children = limit_variants(children, max_variants=3)
            self.logger.debug(f"{self.name}: Generated {len(children)} crossover variants, limited to {len(limited_children)}")
            return limited_children
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed with error: {e}\nTrace: {traceback.format_exc()}")
            return parent_texts[:1] if parent_texts else []

