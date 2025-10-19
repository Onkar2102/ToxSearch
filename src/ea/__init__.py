## @file src/ea/__init__.py
# @brief Evolutionary Algorithm (EA) package for LLM toxicity optimization.
#
# This package provides:
#  - EvolutionEngine: the core EA loop (selection + variation)
#  - run_evolution: driver for one EA generation
#  - TextVariationOperators: concrete mutation operators

# Lazy imports to avoid torch dependency issues
def get_EvolutionEngine():
    """Lazy import of EvolutionEngine to avoid torch dependency issues"""
    from ea.EvolutionEngine import EvolutionEngine
    return EvolutionEngine

def get_run_evolution():
    """Lazy import of run_evolution to avoid torch dependency issues"""
    from ea.RunEvolution import run_evolution
    return run_evolution

def get_create_final_statistics_with_tracker():
    """Lazy import of create_final_statistics_with_tracker to avoid torch dependency issues"""
    from ea.RunEvolution import create_final_statistics_with_tracker
    return create_final_statistics_with_tracker

def get_update_evolution_tracker_with_generation_global():
    """Lazy import of update_evolution_tracker_with_generation_global to avoid torch dependency issues"""
    from ea.RunEvolution import update_evolution_tracker_with_generation_global
    return update_evolution_tracker_with_generation_global

# Mutation operators
def get_LLM_POSAwareSynonymReplacement():
    """Lazy import of LLM_POSAwareSynonymReplacement to avoid torch dependency issues"""
    from ea.synonym_replacement import LLM_POSAwareSynonymReplacement
    return LLM_POSAwareSynonymReplacement

def get_POSAwareAntonymReplacement():
    """Lazy import of POSAwareAntonymReplacement to avoid torch dependency issues"""
    from ea.antonym_replacement import POSAwareAntonymReplacement
    return POSAwareAntonymReplacement

def get_MLMOperator():
    """Lazy import of MLMOperator to avoid torch dependency issues"""
    from ea.mlm_operator import MLMOperator
    return MLMOperator

def get_LLMBasedParaphrasingOperator():
    """Lazy import of LLMBasedParaphrasingOperator to avoid torch dependency issues"""
    from ea.paraphrasing import LLMBasedParaphrasingOperator
    return LLMBasedParaphrasingOperator

def get_StylisticMutator():
    """Lazy import of StylisticMutator to avoid torch dependency issues"""
    from ea.stylistic_mutator import StylisticMutator
    return StylisticMutator

# Back translation operators
def get_LLMBackTranslationHIOperator():
    """Lazy import of LLMBackTranslationHIOperator to avoid torch dependency issues"""
    from ea.back_translation import LLMBackTranslationHIOperator
    return LLMBackTranslationHIOperator

def get_LLMBackTranslationFROperator():
    """Lazy import of LLMBackTranslationFROperator to avoid torch dependency issues"""
    from ea.back_translation import LLMBackTranslationFROperator
    return LLMBackTranslationFROperator

def get_LLMBackTranslationDEOperator():
    """Lazy import of LLMBackTranslationDEOperator to avoid torch dependency issues"""
    from ea.back_translation import LLMBackTranslationDEOperator
    return LLMBackTranslationDEOperator

def get_LLMBackTranslationJAOperator():
    """Lazy import of LLMBackTranslationJAOperator to avoid torch dependency issues"""
    from ea.back_translation import LLMBackTranslationJAOperator
    return LLMBackTranslationJAOperator

def get_LLMBackTranslationZHOperator():
    """Lazy import of LLMBackTranslationZHOperator to avoid torch dependency issues"""
    from ea.back_translation import LLMBackTranslationZHOperator
    return LLMBackTranslationZHOperator

# New mutation operators
def get_NegationOperator():
    """Lazy import of NegationOperator to avoid torch dependency issues"""
    from ea.negation_operator import NegationOperator
    return NegationOperator

def get_TypographicalErrorsOperator():
    """Lazy import of TypographicalErrorsOperator to avoid torch dependency issues"""
    from ea.typographical_errors import TypographicalErrorsOperator
    return TypographicalErrorsOperator

def get_ConceptAdditionOperator():
    """Lazy import of ConceptAdditionOperator to avoid torch dependency issues"""
    from ea.concept_addition import ConceptAdditionOperator
    return ConceptAdditionOperator

def get_InformedEvolutionOperator():
    """Lazy import of InformedEvolutionOperator to avoid torch dependency issues"""
    from ea.InformedEvolution import InformedEvolutionOperator
    return InformedEvolutionOperator

# Crossover operators
def get_SemanticSimilarityCrossover():
    """Lazy import of SemanticSimilarityCrossover to avoid torch dependency issues"""
    from ea.semantic_similarity_crossover import SemanticSimilarityCrossover
    return SemanticSimilarityCrossover

def get_SemanticFusionCrossover():
    """Lazy import of SemanticFusionCrossover to avoid torch dependency issues"""
    from ea.fusion_crossover import SemanticFusionCrossover
    return SemanticFusionCrossover




import logging
logger = logging.getLogger(__name__)

__all__ = [
    # Core evolution functions
    "get_EvolutionEngine",
    "get_run_evolution",
    "get_create_final_statistics_with_tracker",
    "get_update_evolution_tracker_with_generation_global",
    
    # Mutation operators
    "get_LLM_POSAwareSynonymReplacement",
    "get_LLM_POSAwareAntonymReplacement",
    "get_MLMOperator",
    "get_LLMBasedParaphrasingOperator",
    "get_StylisticMutator",
    
    # Back translation operators
    "get_LLMBackTranslationHIOperator",
    "get_LLMBackTranslationFROperator",
    "get_LLMBackTranslationDEOperator",
    "get_LLMBackTranslationJAOperator",
    "get_LLMBackTranslationZHOperator",
    
    # New mutation operators
    "get_NegationOperator",
    "get_TypographicalErrorsOperator",
    "get_ConceptAdditionOperator",
    "get_InformedEvolutionOperator",
    
    # Crossover operators
    "get_SemanticSimilarityCrossover",
    "get_SemanticFusionCrossover",
]