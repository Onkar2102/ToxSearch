"""
operator_helpers.py


This module contains helper functions and common imports used across all
text variation operators in the evolutionary algorithm.

Functions:
    get_generator(): Returns cached LLaMA generator instance
    limit_variants(): Limits number of variants to specified maximum
    get_single_parent_operators(): Returns list of mutation operators
    get_multi_parent_operators(): Returns list of crossover operators
    get_applicable_operators(): Returns operators applicable for given parent count

Version: 1.0
"""

import random
import json
import os
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv

# Get the functions at module level to avoid repeated calls

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()

# Lazy initialization - will be created when first needed
_generator = None

def get_generator():
    """
    Get or create the shared LLaMA text generator instance.
    
    This function implements lazy initialization and caching of the local LLaMA model
    (from models/ directory) to ensure efficient memory usage across all operators that need it.
    
    Returns:
        LlaMaTextGenerator: Cached instance of the LLaMA text generator
        
    Raises:
        ValueError: If model configuration is not found
        FileNotFoundError: If config file is not found
        
    Example:
        >>> generator = get_generator()
        >>> response = generator.generate_response("Hello world")
    """
    global _generator
    if _generator is None:
        from gne import get_LLaMaTextGenerator
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, "config", "modelConfig.yaml")
        LlaMaTextGenerator = get_LLaMaTextGenerator()
        _generator = LlaMaTextGenerator(config_path=config_path, log_file=None)
    return _generator

def limit_variants(variants: List[str], max_variants: int = 3) -> List[str]:
    """
    Limit the number of variants to a maximum value.
    If variants exceed the limit, randomly sample max_variants from them.
    
    Args:
        variants: List of variant strings
        max_variants: Maximum number of variants to return (default: 3)
    
    Returns:
        List of variants limited to max_variants
    """
    if len(variants) <= max_variants:
        return variants
    
    # Randomly sample max_variants from the variants
    selected_variants = random.sample(variants, max_variants)
    return selected_variants


from .VariationOperators import VariationOperator

# Import operators dynamically to avoid circular imports
def get_single_parent_operators(north_star_metric, log_file=None):
    """
    Return list of mutation operators that require only a single parent.
    """
    from .llm_pos_aware_synonym_replacement import LLM_POSAwareSynonymReplacement
    from .mlm_operator import MLMOperator
    from .paraphrasing_operator import LLMBasedParaphrasingOperator
    from .llm_pos_aware_antonym_replacement import LLM_POSAwareAntonymReplacement
    from .stylistic_mutator import StylisticMutator
    from .llm_back_translation_operators import (
        LLMBackTranslationHIOperator, LLMBackTranslationFROperator, LLMBackTranslationDEOperator, LLMBackTranslationJAOperator, LLMBackTranslationZHOperator)
    return [
        LLM_POSAwareSynonymReplacement(log_file=log_file, max_variants=3, num_POS_tags=1),
        MLMOperator(log_file=log_file),
        LLMBasedParaphrasingOperator(north_star_metric, log_file=log_file),
        LLM_POSAwareAntonymReplacement(log_file=log_file, max_variants=3, num_POS_tags=1),
        StylisticMutator(log_file=log_file),
        # Only LLaMA-based back-translation operators
        LLMBackTranslationHIOperator(log_file=log_file),
        LLMBackTranslationFROperator(log_file=log_file),
        LLMBackTranslationDEOperator(log_file=log_file),
        LLMBackTranslationJAOperator(log_file=log_file),
        LLMBackTranslationZHOperator(log_file=log_file),
    ]

def get_multi_parent_operators(north_star_metric="engagement", log_file=None):
    """
    Return list of crossover operators that require multiple parents.
    
    These operators generate variants by combining multiple input prompts.
    They are used for crossover operations in the evolutionary algorithm.
    
    Args:
        north_star_metric (str): The metric to optimize for (e.g., "engagement", "toxicity")
        log_file (str, optional): Path to log file for debugging. Defaults to None.
        
    Returns:
        List[VariationOperator]: List of crossover operators:
            - SemanticSimilarityCrossover: Semantic similarity-based crossover
            - InstructionPreservingCrossover: LLM-based instruction structure preservation with metric optimization
            
    Example:
        >>> operators = get_multi_parent_operators("engagement", "debug.log")
    >>> print(f"Found { len(operators)} crossover operators")
    Found 2 crossover operators
    """
    from .semantic_similarity_crossover import SemanticSimilarityCrossover
    from .instruction_preserving_crossover import InstructionPreservingCrossover
    
    return [
        SemanticSimilarityCrossover(log_file=log_file),
        InstructionPreservingCrossover(north_star_metric=north_star_metric, log_file=log_file)
    ]

def get_applicable_operators(num_parents: int, north_star_metric, log_file=None):
    """
    Return operators applicable for the given number of parents.
    
    This function selects the appropriate set of operators based on the number
    of parent prompts available for variation.
    
    Args:
        num_parents (int): Number of parent prompts available
        north_star_metric (str): The optimization metric for LLMBasedParaphrasingOperator
        log_file (str, optional): Path to log file for debugging. Defaults to None.
        
    Returns:
        List[VariationOperator]: List of applicable operators:
            - If num_parents == 1: Returns mutation operators (single parent)
            - If num_parents > 1: Returns crossover operators (multiple parents)
            
    Example:
        >>> single_ops = get_applicable_operators(1, "engagement_score")
        >>> multi_ops = get_applicable_operators(2, "engagement_score")
        >>> print(f"Single parent: { len(single_ops)}, Multi parent: { len(multi_ops)}")
        Single parent: 4, Multi parent: 3
    """
    if num_parents == 1:
        return get_single_parent_operators(north_star_metric, log_file=log_file)
    return get_multi_parent_operators(log_file=log_file)

load_dotenv()

