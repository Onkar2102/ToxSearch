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




import logging
logger = logging.getLogger(__name__)

__all__ = [
    "get_EvolutionEngine",
    "get_run_evolution",
    "get_create_final_statistics_with_tracker",
    "get_update_evolution_tracker_with_generation_global",
    "get_NegationOperator",
    "get_TypographicalErrorsOperator",
    "get_ConceptAdditionOperator",
    "get_InformedEvolutionOperator",
]