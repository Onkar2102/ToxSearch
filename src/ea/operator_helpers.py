"""
operator_helpers.py

Author: Onkar Shelar os9660@rit.edu

This module contains helper functions and common imports used across all
text variation operators in the evolutionary algorithm.

Functions:
    get_generator(): Returns cached LLaMA generator instance
    limit_variants(): Limits number of variants to specified maximum
    get_single_parent_operators(): Returns list of mutation operators
    get_multi_parent_operators(): Returns list of crossover operators
    get_applicable_operators(): Returns operators applicable for given parent count

Author: EOST CAM LLM Team
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
        # Import here to avoid module-level import issues
        from gne import get_LLaMaTextGenerator
        # Get the project root directory (where config/ folder is located)
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

# Import operator classes for the factory functions
try:
    from ea.VariationOperators import VariationOperator
except Exception:
    # Fallback for direct module execution without package context
    from VariationOperators import VariationOperator

# Import operators dynamically to avoid circular imports
def get_single_parent_operators(north_star_metric, log_file=None):
    """
    Return list of mutation operators that require only a single parent.
    """
    from .pos_aware_synonym_replacement import POSAwareSynonymReplacement
    from .bert_mlm_operator import BertMLMOperator
    from .llm_paraphrasing_operator import LLMBasedParaphrasingOperator
    from .back_translation_hindi import BackTranslationHIOperator
    from .back_translation_french import BackTranslationFROperator
    from .back_translation_german import BackTranslationDEOperator
    from .back_translation_japanese import BackTranslationJAOperator
    from .back_translation_chinese import BackTranslationZHOperator
    from .llm_back_translation_hindi import LLMBackTranslationHIOperator
    from .llm_back_translation_french import LLMBackTranslationFROperator
    from .llm_back_translation_german import LLMBackTranslationDEOperator
    from .llm_back_translation_japanese import LLMBackTranslationJAOperator
    from .llm_back_translation_chinese import LLMBackTranslationZHOperator
    
    return [
        POSAwareSynonymReplacement(log_file=log_file),
        BertMLMOperator(log_file=log_file),
        LLMBasedParaphrasingOperator(north_star_metric, log_file=log_file),
        # Model-based back-translation operators
        BackTranslationHIOperator(log_file=log_file),
        BackTranslationFROperator(log_file=log_file),
        BackTranslationDEOperator(log_file=log_file),
        BackTranslationJAOperator(log_file=log_file),
        BackTranslationZHOperator(log_file=log_file),
        # LLaMA-based back-translation operators
        LLMBackTranslationHIOperator(log_file=log_file),
        LLMBackTranslationFROperator(log_file=log_file),
        LLMBackTranslationDEOperator(log_file=log_file),
        LLMBackTranslationJAOperator(log_file=log_file),
        LLMBackTranslationZHOperator(log_file=log_file),
    ]

def get_multi_parent_operators(log_file=None):
    """
    Return list of crossover operators that require multiple parents.
    
    These operators generate variants by combining multiple input prompts.
    They are used for crossover operations in the evolutionary algorithm.
    
    Args:
        log_file (str, optional): Path to log file for debugging. Defaults to None.
        
    Returns:
        List[VariationOperator]: List of crossover operators:
            - OnePointCrossover: Single-point sentence swapping
            - SemanticSimilarityCrossover: Semantic similarity-based crossover
            - InstructionPreservingCrossover: Instruction structure preservation
            
    Example:
        >>> operators = get_multi_parent_operators("debug.log")
        >>> print(f"Found { len(operators)} crossover operators")
        Found 3 crossover operators
    """
    from .one_point_crossover import OnePointCrossover
    from .semantic_similarity_crossover import SemanticSimilarityCrossover
    from .instruction_preserving_crossover import InstructionPreservingCrossover
    
    return [
        OnePointCrossover(log_file=log_file),
        SemanticSimilarityCrossover(log_file=log_file),
        InstructionPreservingCrossover(log_file=log_file)
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

