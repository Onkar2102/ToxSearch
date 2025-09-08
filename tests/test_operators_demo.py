#!/usr/bin/env python3
"""
Simple demonstration of variation operators with example prompts.
This script shows how each operator creates variants from input prompts.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_limit_variants():
    """Test the limit_variants helper function."""
    from ea.TextVariationOperators import limit_variants
    
    print("=" * 60)
    print("TESTING LIMIT_VARIANTS FUNCTION")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        (["variant1", "variant2"], 3, "Within limit"),
        (["v1", "v2", "v3", "v4", "v5"], 3, "Over limit"),
        ([], 3, "Empty list"),
        (["single"], 3, "Single variant")
    ]
    
    for variants, max_variants, description in test_cases:
        result = limit_variants(variants, max_variants)
        print(f"\n{description}:")
        print(f"  Input: {variants}")
        print(f"  Max: {max_variants}")
        print(f"  Output: {result}")
        print(f"  Count: {len(result)}")


def test_mutation_operators():
    """Test mutation operators with example prompts."""
    from ea.TextVariationOperators import (
        POSAwareSynonymReplacement,
        BertMLMOperator,
        LLMBasedParaphrasingOperator,
        BackTranslationOperator
    )
    
    print("\n" + "=" * 60)
    print("TESTING MUTATION OPERATORS")
    print("=" * 60)
    
    # Example prompts
    example_prompts = [
        "Write a story about a brave knight.",
        "Explain how photosynthesis works in plants.",
        "Create a recipe for chocolate cake."
    ]
    
    operators = [
        ("POSAwareSynonymReplacement", POSAwareSynonymReplacement),
        ("BertMLMOperator", BertMLMOperator),
        ("LLMBasedParaphrasingOperator", LLMBasedParaphrasingOperator),
        ("BackTranslationOperator", BackTranslationOperator)
    ]
    
    for prompt in example_prompts:
        print(f"\n{'='*20} PROMPT: {prompt} {'='*20}")
        
        for op_name, op_class in operators:
            print(f"\n--- {op_name} ---")
            try:
                # Initialize operator
                if op_name == "LLMBasedParaphrasingOperator":
                    operator = op_class("mock_metric", log_file=None)
                else:
                    operator = op_class(log_file=None)
                
                # Apply operator
                variants = operator.apply(prompt)
                
                print(f"Generated {len(variants)} variants:")
                for i, variant in enumerate(variants, 1):
                    print(f"  {i}. {variant}")
                    
            except Exception as e:
                print(f"  Error: {str(e)[:100]}...")


def test_crossover_operators():
    """Test crossover operators with example prompts."""
    from ea.TextVariationOperators import (
        OnePointCrossover,
        SemanticSimilarityCrossover,
        InstructionPreservingCrossover
    )
    
    print("\n" + "=" * 60)
    print("TESTING CROSSOVER OPERATORS")
    print("=" * 60)
    
    # Example parent pairs
    parent_pairs = [
        (["Write a story about a brave knight.", "Create a tale about a magical dragon."]),
        (["Explain how photosynthesis works.", "Describe the process of cellular respiration."]),
        (["Make a chocolate cake recipe.", "Prepare a vanilla ice cream recipe."])
    ]
    
    operators = [
        ("OnePointCrossover", OnePointCrossover),
        ("SemanticSimilarityCrossover", SemanticSimilarityCrossover),
        ("InstructionPreservingCrossover", InstructionPreservingCrossover)
    ]
    
    for i, parents in enumerate(parent_pairs, 1):
        print(f"\n{'='*20} PARENT PAIR {i} {'='*20}")
        print(f"Parent 1: {parents[0]}")
        print(f"Parent 2: {parents[1]}")
        
        for op_name, op_class in operators:
            print(f"\n--- {op_name} ---")
            try:
                # Initialize operator
                operator = op_class(log_file=None)
                
                # Apply operator
                variants = operator.apply(parents)
                
                print(f"Generated {len(variants)} variants:")
                for j, variant in enumerate(variants, 1):
                    print(f"  {j}. {variant}")
                    
            except Exception as e:
                print(f"  Error: {str(e)[:100]}...")


def test_operator_selection():
    """Test operator selection functions."""
    from ea.TextVariationOperators import (
        get_single_parent_operators,
        get_multi_parent_operators,
        get_applicable_operators
    )
    
    print("\n" + "=" * 60)
    print("TESTING OPERATOR SELECTION")
    print("=" * 60)
    
    # Test single parent operators
    print("\n--- Single Parent Operators ---")
    try:
        single_ops = get_single_parent_operators("mock_metric", log_file=None)
        print(f"Found {len(single_ops)} single parent operators:")
        for op in single_ops:
            print(f"  - {op.name} ({op.operator_type})")
    except Exception as e:
        print(f"Error: {str(e)[:100]}...")
    
    # Test multi parent operators
    print("\n--- Multi Parent Operators ---")
    try:
        multi_ops = get_multi_parent_operators(log_file=None)
        print(f"Found {len(multi_ops)} multi parent operators:")
        for op in multi_ops:
            print(f"  - {op.name} ({op.operator_type})")
    except Exception as e:
        print(f"Error: {str(e)[:100]}...")
    
    # Test applicable operators
    print("\n--- Applicable Operators ---")
    test_cases = [
        (1, "Single parent"),
        (2, "Two parents"),
        (3, "Three parents"),
        (0, "Zero parents")
    ]
    
    for num_parents, description in test_cases:
        try:
            ops = get_applicable_operators(num_parents, "mock_metric", log_file=None)
            print(f"{description} ({num_parents}): {len(ops)} operators")
            for op in ops:
                print(f"  - {op.name} ({op.operator_type})")
        except Exception as e:
            print(f"{description}: Error - {str(e)[:100]}...")


def main():
    """Main function to run all tests."""
    print("EOST CAM LLM - OPERATOR DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates how variation operators work")
    print("with example prompts.")
    
    # Run tests
    test_limit_variants()
    test_mutation_operators()
    test_crossover_operators()
    test_operator_selection()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Note: Some operators may fail due to missing dependencies")
    print("(BERT models, OpenAI API keys, etc.)")
    print("This is expected behavior for demonstration purposes.")


if __name__ == "__main__":
    main()