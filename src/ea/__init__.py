## @file src/ea/__init__.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Evolutionary Algorithm (EA) package for LLM toxicity optimization.
#
# This package provides:
#  - EvolutionEngine: the core EA loop (selection + variation)
#  - run_evolution: driver for one EA generation
#  - TextVariationOperators: concrete mutation operators
#  - get_applicable_operators: helper to pick operators based on parent count

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

def get_applicable_operators():
    """Lazy import of get_applicable_operators to avoid torch dependency issues"""
    from ea.TextVariationOperators import get_applicable_operators
    return get_applicable_operators

import logging
logger = logging.getLogger(__name__)

__all__ = [
    "get_EvolutionEngine",
    "get_run_evolution",
    "get_create_final_statistics_with_tracker",
    "get_applicable_operators",
]