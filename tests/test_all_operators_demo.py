#!/usr/bin/env python3
"""
Comprehensive operator demonstration using the newly separated operator files.

This module demonstrates all operators working with the separated file structure,
replacing the old TextVariationOperators.py approach.
"""

import sys
import os

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def demo_mutation_operators():
    """Demonstrate mutation operators from separated files."""
    print("\n🧬 MUTATION OPERATORS (from separated files):")
    print("-" * 50)
    
    test_prompt = "Write a story about a brave knight who faces adversity."
    
    # Import and test individual mutation operators
    try:
        from ea.pos_aware_synonym_replacement import POSAwareSynonymReplacement
        operator = POSAwareSynonymReplacement()
        variants = operator.apply(test_prompt)
        print(f"\n✓ POSAwareSynonymReplacement (BERT): {len(variants)} variants")
        for i, variant in enumerate(variants[:2], 1):
            print(f"  {i}. {variant[:70]}...")
    except Exception as e:
        print(f"✗ POSAwareSynonymReplacement (BERT): {str(e)[:60]}...")
    
    try:
        from ea.llm_pos_aware_synonym_replacement import LLM_POSAwareSynonymReplacement
        operator = LLM_POSAwareSynonymReplacement(log_file=None, max_variants=3, num_POS_tags=1)
        variants = operator.apply(test_prompt)
        print(f"\n✓ LLM_POSAwareSynonymReplacement: {len(variants)} variants")
        for i, variant in enumerate(variants[:2], 1):
            print(f"  {i}. {variant[:70]}...")
    except Exception as e:
        print(f"✗ LLM_POSAwareSynonymReplacement: {str(e)[:60]}...")
    
    try:
        from ea.bert_mlm_operator import BertMLMOperator
        operator = BertMLMOperator()
        variants = operator.apply(test_prompt)
        print(f"\n✓ BertMLMOperator: {len(variants)} variants")
        for i, variant in enumerate(variants[:2], 1):
            print(f"  {i}. {variant[:70]}...")
    except Exception as e:
        print(f"✗ BertMLMOperator: {str(e)[:60]}...")
    
    try:
        from ea.llm_paraphrasing_operator import LLMBasedParaphrasingOperator
        operator = LLMBasedParaphrasingOperator("engagement")
        variants = operator.apply(test_prompt)
        print(f"\n✓ LLMBasedParaphrasingOperator: {len(variants)} variants")
        for i, variant in enumerate(variants[:1], 1):
            print(f"  {i}. {variant[:70]}...")
    except Exception as e:
        print(f"✗ LLMBasedParaphrasingOperator: {str(e)[:60]}...")


def demo_back_translation_operators():
    """Demonstrate back-translation operators from separated files."""
    print("\n🌍 BACK-TRANSLATION OPERATORS (from separated files):")
    print("-" * 50)
    
    test_prompt = "Write a story about a brave knight who faces adversity."
    
    # Helsinki-NLP model-based operators
    helsinki_operators = [
        ("BackTranslationFROperator", "ea.back_translation_french"),
        ("BackTranslationDEOperator", "ea.back_translation_german"),
        ("BackTranslationJAOperator", "ea.back_translation_japanese"),
        ("BackTranslationZHOperator", "ea.back_translation_chinese"),
        ("BackTranslationHIOperator", "ea.back_translation_hindi"),
    ]
    
    print("\n📦 Helsinki-NLP Models:")
    for op_name, module_name in helsinki_operators:
        try:
            module = __import__(module_name, fromlist=[op_name])
            operator_class = getattr(module, op_name)
            operator = operator_class()
            variants = operator.apply(test_prompt)
            print(f"  ✓ {op_name}: {len(variants)} variants")
        except Exception as e:
            print(f"  ✗ {op_name}: Model not available - {str(e)[:40]}...")
    
    # LLaMA-based operators
    llma_operators = [
        ("LLMBackTranslationFROperator", "ea.llm_back_translation_french"),
        ("LLMBackTranslationDEOperator", "ea.llm_back_translation_german"),
        ("LLMBackTranslationJAOperator", "ea.llm_back_translation_japanese"),
        ("LLMBackTranslationZHOperator", "ea.llm_back_translation_chinese"),
        ("LLMBackTranslationHIOperator", "ea.llm_back_translation_hindi"),
    ]
    
    print("\n🤖 LLaMA-Based Models:")
    for op_name, module_name in llma_operators:
        try:
            module = __import__(module_name, fromlist=[op_name])
            operator_class = getattr(module, op_name)
            operator = operator_class()
            variants = operator.apply(test_prompt)
            print(f"  ✓ {op_name}: {len(variants)} variants")
        except Exception as e:
            print(f"  ✗ {op_name}: Generator not available - {str(e)[:40]}...")


def demo_crossover_operators():
    """Demonstrate crossover operators from separated files."""
    print("\n🔄 CROSSOVER OPERATORS (from separated files):")
    print("-" * 50)
    
    parent_prompts = [
        "Write a story about a brave knight who faces adversity.",
        "Create a tale about a magical dragon and its adventures."
    ]
    
    crossover_operators = [
        ("OnePointCrossover", "ea.one_point_crossover"),
        ("SemanticSimilarityCrossover", "ea.semantic_similarity_crossover"),
        ("InstructionPreservingCrossover", "ea.instruction_preserving_crossover"),
    ]
    
    for op_name, module_name in crossover_operators:
        try:
            module = __import__(module_name, fromlist=[op_name])
            operator_class = getattr(module, op_name)
            operator = operator_class()
            variants = operator.apply(parent_prompts)
            print(f"\n✓ {op_name}: {len(variants)} variants")
            for i, variant in enumerate(variants[:1], 1):
                print(f"  {i}. {variant[:70]}...")
        except Exception as e:
            print(f"\n✗ {op_name}: {str(e)[:60]}...")


def demo_helper_functions():
    """Demonstrate operator helper functions."""
    print("\n🛠️ HELPER FUNCTIONS (from separated files):")
    print("-" * 50)
    
    try:
        from ea.operator_helpers import get_single_parent_operators
        mutation_ops = get_single_parent_operators("engagement")
        print(f"✓ get_single_parent_operators: {len(mutation_ops)} operators")
        for op in mutation_ops[:3]:  # Show first 3
            print(f"  - {op.name}")
    except Exception as e:
        print(f"✗ get_single_parent_operators: {str(e)[:60]}...")
    
    try:
        from ea.operator_helpers import get_multi_parent_operators
        crossover_ops = get_multi_parent_operators()
        print(f"✓ get_multi_parent_operators: {len(crossover_ops)} operators")
        for op in crossover_ops:
            print(f"  - {op.name}")
    except Exception as e:
        print(f"✗ get_multi_parent_operators: {str(e)[:60]}...")
    
    try:
        from ea.operator_helpers import get_applicable_operators
        applicable_single = get_applicable_operators(1, "engagement")
        applicable_multi = get_applicable_operators(2, "engagement")
        print(f"✓ get_applicable_operators: {len(applicable_single)} single-parent, {len(applicable_multi)} multi-parent")
    except Exception as e:
        print(f"✗ get_applicable_operators: {str(e)[:60]}...")


def demo_comparison():
    """Compare old and new approaches."""
    print("\n📊 COMPARISON: Old vs New Approach")
    print("-" * 50)
    
    print("OLD APPROACH (TextVariationOperators.py):")
    print("  ❌ Single file with all operators (~1000+ lines)")
    print("  ❌ Hard to maintain and extend")
    print("  ❌ Difficult to test individual operators")
    print("  ❌ Circular import risks")
    print("  ❌ Import all dependencies even if using one operator")
    
    print("\nNEW APPROACH (Separated files):")
    print("  ✅ Each operator in its own file")
    print("  ✅ Easy to maintain and extend")
    print("  ✅ Individual operator testing")
    print("  ✅ Modular imports (only import what you need)")
    print("  ✅ Clear separation of concerns")
    print("  ✅ Better organization and documentation")
    
    print("\nFILES CREATED:")
    print("  📁 operator_helpers.py - Helper functions and utilities")
    print("  📁 base_operators.py - Base classes for operator families")
    print("  📁 pos_aware_synonym_replacement.py - POS-aware synonym replacement")
    print("  📁 bert_mlm_operator.py - BERT MLM word replacement")
    print("  📁 llm_paraphrasing_operator.py - OpenAI paraphrasing")
    print("  📁 back_translation_*.py - 5 Helsinki-NLP back-translation operators")
    print("  📁 llm_back_translation_*.py - 5 LLaMA back-translation operators")
    print("  📁 *_crossover.py - 3 crossover operators")


def main():
    """Run comprehensive demo of all separated operators."""
    print("🚀 EOST CAM LLM - SEPARATED OPERATORS DEMO")
    print("=" * 60)
    print("Testing the new modular operator architecture...")
    
    demo_mutation_operators()
    demo_back_translation_operators()
    demo_crossover_operators()
    demo_helper_functions()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETE")
    print("\nAll operators now use the separated file architecture!")
    print("The monolith has been broken down into maintainable modules.")
    print("Each operator can now be imported and tested independently.")


if __name__ == "__main__":
    main()
