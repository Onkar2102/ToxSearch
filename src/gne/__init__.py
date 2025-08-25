## @file src/gne/__init__.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Generative Neural Engine (GNE) package for LLM integration and moderation.
#
# This package provides:
#  - LLaMaTextGenerator: LLaMA model integration with memory management
#  - hybrid_moderation: Hybrid moderation using Google and OpenAI APIs

# Lazy imports to prevent circular import issues
def get_LLaMaTextGenerator():
    """Lazy import of LLaMaTextGenerator to avoid circular imports"""
    from gne.LLaMaTextGenerator import LlaMaTextGenerator
    return LlaMaTextGenerator

def get_run_moderation_on_population():
    """Lazy import of run_moderation_on_population to avoid circular imports"""
    from gne.hybrid_moderation import run_moderation_on_population
    return run_moderation_on_population

def get_hybrid_moderation_cleanup():
    """Lazy import of _cleanup_thread_pool to avoid circular imports"""
    from gne.hybrid_moderation import _cleanup_thread_pool
    return _cleanup_thread_pool

__all__ = [
    "get_LLaMaTextGenerator",
    "get_run_moderation_on_population",
    "get_hybrid_moderation_cleanup",
]
