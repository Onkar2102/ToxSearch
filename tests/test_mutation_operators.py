#!/usr/bin/env python3
"""
Test suite for mutation operators (single parent operators).

This module tests all mutation operators to ensure they work correctly
after the separation into individual files.
"""

import sys
import os
import pytest

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMutationOperators:
    """Test class for mutation operators."""
    
    def setup_method(self):
        """Setup test environment."""
        self.test_prompt = "Write a story about a brave knight who faces adversity."
        self.north_star_metric = "engagement"
        
    def test_pos_aware_synonym_replacement(self):
        """Test POS-aware synonym replacement operator."""
        from ea.pos_aware_synonym_replacement import POSAwareSynonymReplacement
        
        operator = POSAwareSynonymReplacement()
        
        variants = operator.apply(self.test_prompt)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3  # Should be limited to 3
        assert all(isinstance(variant, str) for variant in variants)
        
        # Check that at least one variant is different from original
        variants_normalized = [v.lower().strip() for v in variants]
        original_normalized = self.test_prompt.lower().strip()
        assert any(variant != original_normalized for variant in variants_normalized)
        
        print(f"POSAwareSynonymReplacement: Generated {len(variants)} variants")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:60]}...")
    
    def test_bert_mlm_operator(self):
        """Test BERT MLM operator."""
        from ea.bert_mlm_operator import BertMLMOperator
        
        operator = BertMLMOperator()
        
        variants = operator.apply(self.test_prompt)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3  # Should be limited to 3
        assert all(isinstance(variant, str) for variant in variants)
        
        print(f"BertMLMOperator: Generated {len(variants)} variants")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:60]}...")
    
    def test_llm_paraphrasing_operator(self):
        """Test LLM paraphrasing operator."""
        from ea.llm_paraphrasing_operator import LLMBasedParaphrasingOperator
        
        operator = LLMBasedParaphrasingOperator(self.north_star_metric)
        
        variants = operator.apply(self.test_prompt)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3  # Should be limited to 3
        assert all(isinstance(variant, str) for variant in variants)
        
        print(f"LLMBasedParaphrasingOperator: Generated {len(variants)} variants")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:60]}...")
    
    def test_back_translation_hindi(self):
        """Test Hindi back-translation operator."""
        from ea.back_translation_hindi import BackTranslationHIOperator
        
        operator = BackTranslationHIOperator()
        
        variants = operator.apply(self.test_prompt)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3  # Should be limited to 3
        assert all(isinstance(variant, str) for variant in variants)
        
        print(f"BackTranslationHIOperator: Generated {len(variants)} variants")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:60]}...")
    
    def test_llm_back_translation_french(self):
        """Test LLaMA-based French back-translation operator."""
        from ea.llm_back_translation_french import LLMBackTranslationFROperator
        
        operator = LLMBackTranslationFROperator()
        
        variants = operator.apply(self.test_prompt)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3  # Should be limited to 3
        assert all(isinstance(variant, str) for variant in variants)
        
        print(f"LLMBackTranslationFROperator: Generated {len(variants)} variants")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:60]}...")
    
    def test_llm_pos_aware_synonym_replacement(self):
        """Test LLM POS-aware synonym replacement operator."""
        from ea.llm_pos_aware_synonym_replacement import LLM_POSAwareSynonymReplacement
        
        operator = LLM_POSAwareSynonymReplacement(max_variants=3, num_POS_tags=2, seed=42)
        
        variants = operator.apply(self.test_prompt)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3  # Should be limited to 3
        assert all(isinstance(variant, str) for variant in variants)
        
        # Test that operator has expected attributes
        assert hasattr(operator, 'max_variants')
        assert hasattr(operator, 'num_POS_tags')
        assert hasattr(operator, 'seed')
        assert operator.name == "LLM_POSAwareSynonymReplacement"
        assert operator.operator_type == "mutation"
        
        # Test POS detection functionality
        pos_info = operator.get_pos_info(self.test_prompt)
        assert isinstance(pos_info, dict)
        assert 'detected_pos_types' in pos_info
        assert 'selected_pos_types' in pos_info
        
        print(f"LLM_POSAwareSynonymReplacement: Generated {len(variants)} variants")
        print(f"  POS types detected: {len(pos_info['detected_pos_types'])}")
        print(f"  POS types selected: {len(pos_info['selected_pos_types'])}")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant[:60]}...")


def run_mutation_tests():
    """Run all mutation operator tests and print results."""
    print("Testing Mutation Operators")
    print("=" * 50)
    
    test_instance = TestMutationOperators()
    test_instance.setup_method()
    
    tests = [
        ("POSAwareSynonymReplacement", test_instance.test_pos_aware_synonym_replacement),
        ("BertMLMOperator", test_instance.test_bert_mlm_operator),
        ("LLMBasedParaphrasingOperator", test_instance.test_llm_paraphrasing_operator),
        ("BackTranslationHIOperator", test_instance.test_back_translation_hindi),
        ("LLMBackTranslationFROperator", test_instance.test_llm_back_translation_french),
        ("LLM_POSAwareSynonymReplacement", test_instance.test_llm_pos_aware_synonym_replacement),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nS {test_name}:")
        print("-" * 30)
        try:
            test_func()
            print(f"S {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: FAILED - {e}")
    
    print(f"\nMutation Operator Tests: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    success = run_mutation_tests()
    sys.exit(0 if success else 1)
