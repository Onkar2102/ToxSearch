#!/usr/bin/env python3
"""
Pytest-based test suite for the separated operator architecture.

This file provides comprehensive pytest tests for all operators
using the new modular file structure.
"""

import sys
import os
import pytest

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "Write a story about a brave knight who faces adversity."


@pytest.fixture
def sample_prompts():
    """Sample prompts for crossover testing."""
    return [
        "Write a story about a brave knight who faces adversity.",
        "Create a tale about a magical dragon and its adventures."
    ]


class TestMutationOperatorsSeparated:
    """Test mutation operators from separated files."""
    
    def test_pos_aware_synonym_replacement_import_and_instantiation(self):
        """Test importing and instantiating POSAwareSynonymReplacement."""
        from ea.pos_aware_synonym_replacement import POSAwareSynonymReplacement
        
        operator = POSAwareSynonymReplacement()
        
        assert operator.name == "POSAwareSynonymReplacement"
        assert operator.operator_type == "mutation"
        assert "BERT-based synonym replacement" in operator.description
    
    def test_pos_aware_synonym_replacement_functionality(self, sample_text):
        """Test POSAwareSynonymReplacement functionality."""
        from ea.pos_aware_synonym_replacement import POSAwareSynonymReplacement
        
        operator = POSAwareSynonymReplacement()
        variants = operator.apply(sample_text)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3
        assert all(isinstance(variant, str) for variant in variants)
    
    def test_bert_mlm_operator_import_and_instantiation(self):
        """Test importing and instantiating BertMLMOperator."""
        from ea.bert_mlm_operator import BertMLMOperator
        
        operator = BertMLMOperator()
        
        assert operator.name == "BertMLM"
        assert operator.operator_type == "mutation"
        assert "BERT MLM" in operator.description
    
    def test_bert_mlm_operator_functionality(self, sample_text):
        """Test BertMLMOperator functionality."""
        from ea.bert_mlm_operator import BertMLMOperator
        
        operator = BertMLMOperator()
        variants = operator.apply(sample_text)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3
        assert all(isinstance(variant, str) for variant in variants)
    
    def test_llm_paraphrasing_operator_import_and_instantiation(self):
        """Test importing and instantiating LLMBasedParaphrasingOperator."""
        from ea.llm_paraphrasing_operator import LLMBasedParaphrasingOperator
        
        operator = LLMBasedParaphrasingOperator("engagement")
        
        assert operator.name == "LLMBasedParaphrasing"
        assert operator.operator_type == "mutation"
        assert "OpenAI LLM" in operator.description
        assert operator.north_star_metric == "engagement"
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    def test_llm_paraphrasing_operator_functionality(self, sample_text):
        """Test LLMBasedParaphrasingOperator functionality."""
        from ea.llm_paraphrasing_operator import LLMBasedParaphrasingOperator
        
        operator = LLMBasedParaphrasingOperator("engagement")
        variants = operator.apply(sample_text)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3
        assert all(isinstance(variant, str) for variant in variants)


class TestBackTranslationOperatorsSeparated:
    """Test back-translation operators from separated files."""
    
    def test_back_translation_hindi_operator_import_and_instantiation(self):
        """Test importing and instantiating BackTranslationHIOperator."""
        from ea.back_translation_hindi import BackTranslationHIOperator
        
        operator = BackTranslationHIOperator()
        
        assert operator.name == "BackTranslation_HI"
        assert operator.operator_type == "mutation"
        assert "Hindi back-translation" in operator.description
    
    def test_back_translation_french_operator_import_and_instantiation(self):
        """Test importing and instantiating BackTranslationFROperator."""
        from ea.back_translation_french import BackTranslationFROperator
        
        operator = BackTranslationFROperator()
        
        assert operator.name == "BackTranslation_FR"
        assert operator.operator_type == "mutation"
        assert "French back-translation" in operator.description
    
    def test_llm_back_translation_hindi_operator_import_and_instantiation(self):
        """Test importing and instantiating LLMBackTranslationHIOperator."""
        from ea.llm_back_translation_hindi import LLMBackTranslationHIOperator
        
        operator = LLMBackTranslationHIOperator()
        
        assert operator.name == "LLMBackTranslation_HI"
        assert operator.operator_type == "mutation"
        assert "LLaMA-based" in operator.description
    
    def test_back_translation_operators_inheritance(self):
        """Test that back-translation operators inherit from correct base classes."""
        from ea.back_translation_french import BackTranslationFROperator
        from ea.llm_back_translation_french import LLMBackTranslationFROperator
        from ea.base_operators import _GenericBackTranslationOperator, _GenericLLMBackTranslationOperator
        
        # Test Helsinki-NLP operator inheritance
        helsinki_op = BackTranslationFROperator()
        assert isinstance(helsinki_op, _GenericBackTranslationOperator)
        
        # Test LLaMA operator inheritance
        llma_op = LLMBackTranslationFROperator()
        assert isinstance(llma_op, _GenericLLMBackTranslationOperator)


class TestCrossoverOperatorsSeparated:
    """Test crossover operators from separated files."""
    
    def test_one_point_crossover_import_and_instantiation(self):
        """Test importing and instantiating OnePointCrossover."""
        from ea.one_point_crossover import OnePointCrossover
        
        operator = OnePointCrossover()
        
        assert operator.name == "OnePointCrossover"
        assert operator.operator_type == "crossover"
        assert "matching-position sentences" in operator.description
    
    def test_one_point_crossover_functionality(self, sample_prompts):
        """Test OnePointCrossover functionality."""
        from ea.one_point_crossover import OnePointCrossover
        
        operator = OnePointCrossover()
        variants = operator.apply(sample_prompts)
        
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert len(variants) <= 3
        assert all(isinstance(variant, str) for variant in variants)
    
    def test_semantic_similarity_crossover_import_and_instantiation(self):
        """Test importing and instantiating SemanticSimilarityCrossover."""
        from ea.semantic_similarity_crossover import SemanticSimilarityCrossover
        
        operator = SemanticSimilarityCrossover()
        
        assert operator.name == "SemanticSimilarityCrossover"
        assert operator.operator_type == "crossover"
        assert "semantically similar" in operator.description
    
    def test_instruction_preserving_crossover_import_and_instantiation(self):
        """Test importing and instantiating InstructionPreservingCrossover."""
        from ea.instruction_preserving_crossover import InstructionPreservingCrossover
        
        operator = InstructionPreservingCrossover()
        
        assert operator.name == "InstructionPreservingCrossover"
        assert operator.operator_type == "crossover"
        assert "instruction head" in operator.description
    
    def test_crossover_operators_error_handling(self):
        """Test crossover operators handle edge cases properly."""
        from ea.one_point_crossover import OnePointCrossover
        
        operator = OnePointCrossover()
        
        # Test with empty list
        result = operator.apply([])
        assert isinstance(result, list)
        assert len(result) == 0
        
        # Test with single element
        result = operator.apply(["single prompt"])
        assert isinstance(result, list)
        assert len(result) == 1


class TestOperatorHelpersSeparated:
    """Test operator helper functions from separated files."""
    
    def test_limit_variants_function(self):
        """Test limit_variants utility function."""
        from ea.operator_helpers import limit_variants
        
        # Test normal case
        variants = ["v1", "v2", "v3", "v4", "v5"]
        limited = limit_variants(variants, max_variants=3)
        assert len(limited) == 3
        assert all(var in variants for var in limited)
        
        # Test when variants are fewer than limit
        limited = limit_variants(["v1", "v2"], max_variants=3)
        assert len(limited) == 2
        
        # Test empty list
        limited = limit_variants([], max_variants=3)
        assert len(limited) == 0
    
    def test_get_single_parent_operators(self):
        """Test get_single_parent_operators function."""
        from ea.operator_helpers import get_single_parent_operators
        
        operators = get_single_parent_operators("test_metric")
        
        assert isinstance(operators, list)
        assert len(operators) > 0
        
        for operator in operators:
            assert operator.operator_type == "mutation"
            assert hasattr(operator, 'name')
            assert hasattr(operator, 'operator_type')
            assert hasattr(operator, 'description')
    
    def test_get_multi_parent_operators(self):
        """Test get_multi_parent_operators function."""
        from ea.operator_helpers import get_multi_parent_operators
        
        operators = get_multi_parent_operators()
        
        assert isinstance(operators, list)
        assert len(operators) > 0
        
        for operator in operators:
            assert operator.operator_type == "crossover"
            assert hasattr(operator, 'name')
            assert hasattr(operator, 'operator_type')
            assert hasattr(operator, 'description')
    
    def test_get_applicable_operators(self):
        """Test get_applicable_operators function."""
        from ea.operator_helpers import get_applicable_operators
        
        # Test with single parent
        single_ops = get_applicable_operators(1, "test_metric")
        assert isinstance(single_ops, list)
        assert all(op.operator_type == "mutation" for op in single_ops)
        
        # Test with multiple parents
        multi_ops = get_applicable_operators(2, "test_metric")
        assert isinstance(multi_ops, list)
        assert all(op.operator_type == "crossover" for op in multi_ops)


if __name__ == "__main__":
    pytest.main([__file__])
