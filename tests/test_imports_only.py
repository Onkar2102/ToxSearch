#!/usr/bin/env python3
"""
Quick test to verify imports are working correctly.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from ea.TextVariationOperators import limit_variants
        print("✅ limit_variants imported successfully")
    except Exception as e:
        print(f"❌ limit_variants import failed: {e}")
    
    try:
        from ea.TextVariationOperators import (
            POSAwareSynonymReplacement,
            BertMLMOperator,
            LLMBasedParaphrasingOperator,
            BackTranslationOperator
        )
        print("✅ Mutation operators imported successfully")
    except Exception as e:
        print(f"❌ Mutation operators import failed: {e}")
    
    try:
        from ea.TextVariationOperators import (
            OnePointCrossover,
            SemanticSimilarityCrossover,
            InstructionPreservingCrossover
        )
        print("✅ Crossover operators imported successfully")
    except Exception as e:
        print(f"❌ Crossover operators import failed: {e}")
    
    try:
        from ea.TextVariationOperators import (
            get_single_parent_operators,
            get_multi_parent_operators,
            get_applicable_operators
        )
        print("✅ Operator selection functions imported successfully")
    except Exception as e:
        print(f"❌ Operator selection functions import failed: {e}")
    
    print("\nAll imports tested!")

if __name__ == "__main__":
    test_imports()
