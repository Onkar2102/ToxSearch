#!/usr/bin/env python3
"""
Test suite for crossover operators (multiple parent operators).

This module tests all crossover operators to ensure they work correctly
after the separation into individual files.
"""

import sys
import os
import pytest

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCrossoverOperators:
    """Test class for crossover operators."""
    
    def setup_method(self):
        """Setup test environment."""
        self.parent_prompts = [
            "Write a story about a brave knight who faces adversity.",
            "Create a tale about a magical dragon and its adventures."
        ]
        self.simple_prompts = [
            "Tell me about a hero.",
            "How does magic work?"
        ]
        
    def test_one_point_crossover(self):
        """Test one-point crossover operator."""
        from ea.one_point_crossover import OnePointCrossover
        
        operator = OnePointCrossover()
        
        variants = operator.apply(self.parent_prompts)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3  # Should be limited to 3
        assert all(isinstance(variant, str) for variant in variants)
        
        # Check that variants contain parts from both parents
        # (This is a basic check - more sophisticated verification would be complex)
        print(f"OnePointCrossover: Generated {len(variants)} variants")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:80]}...")
    
    def test_semantic_similarity_crossover(self):
        """Test semantic similarity crossover operator."""
        from ea.semantic_similarity_crossover import SemanticSimilarityCrossover
        
        operator = SemanticSimilarityCrossover()
        
        variants = operator.apply(self.parent_prompts)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3  # Should be limited to 3
        assert all(isinstance(variant, str) for variant in variants)
        
        print(f"SemanticSimilarityCrossover: Generated {len(variants)} variants")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:80]}...")
    
    def test_instruction_preserving_crossover(self):
        """Test instruction preserving crossover operator."""
        from ea.instruction_preserving_crossover import InstructionPreservingCrossover
        
        operator = InstructionPreservingCrossover()
        
        variants = operator.apply(self.parent_prompts)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3  # Should be limited to 3
        assert all(isinstance(variant, str) for variant in variants)
        
        print(f"InstructionPreservingCrossover: Generated {len(variants)} variants")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:80]}...")
    
    def test_one_point_crossover_simple(self):
        """Test one-point crossover with simpler prompts."""
        from ea.one_point_crossover import OnePointCrossover
        
        operator = OnePointCrossover()
        
        variants = operator.apply(self.simple_prompts)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3  # Should be limited to 3
        assert all(isinstance(variant, str) for variant in variants)
        
        print(f"OnePointCrossover (simple): Generated {len(variants)} variants")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:80]}...")
    
    def test_operator_error_handling(self):
        """Test error handling in crossover operators."""
        from ea.one_point_crossover import OnePointCrossover
        
        operator = OnePointCrossover()
        
        # Test with insufficient parents
        variants = operator.apply(["Only one parent"])
        
        assert isinstance(variants, list)
        assert len(variants) == 1
        assert isinstance(variants[0], str)
        
        # Test with empty list
        variants = operator.apply([])
        
        assert isinstance(variants, list)
        assert len(variants) == 0
        
        print("Error handling tests: PASSED")


def run_crossover_tests():
    """Run all crossover operator tests and print results."""
    print("Testing Crossover Operators")
    print("=" * 50)
    
    test_instance = TestCrossoverOperators()
    test_instance.setup_method()
    
    tests = [
        ("OnePointCrossover", test_instance.test_one_point_crossover),
        ("SemanticSimilarityCrossover", test_instance.test_semantic_similarity_crossover),
        ("InstructionPreservingCrossover", test_instance.test_instruction_preserving_crossover),
        ("OnePointCrossoverSimple", test_instance.test_one_point_crossover_simple),
        ("ErrorHandling", test_instance.test_operator_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nC {test_name}:")
        print("-" * 30)
        try:
            test_func()
            print(f"S {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: FAILED - {e}")
    
    print(f"\nCrossover Operator Tests: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    success = run_crossover_tests()
    sys.exit(0 if success else 1)
