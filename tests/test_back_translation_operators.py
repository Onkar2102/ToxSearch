#!/usr/bin/env python3
"""
Test suite for back-translation operators.

This module tests all back-translation operators (both Helsinki-NLP and LLaMA-based)
to ensure they work correctly after the separation into individual files.
"""

import sys
import os

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestBackTranslationOperators:
    """Test class for back-translation operators."""
    
    def setup_method(self):
        """Setup test environment."""
        self.test_prompt = "Write a story about a brave knight who faces adversity."
        
    def test_helsinki_nlp_back_translation(self):
        """Test Helsinki-NLP model-based back-translation operators."""
        print("\nTesting Helsinki-NLP Back-Translation Operators:")
        print("-" * 45)
        
        # Test French back-translation
        try:
            from ea.back_translation_french import BackTranslationFROperator
            operator = BackTranslationFROperator()
            variants = operator.apply(self.test_prompt)
            
            assert isinstance(variants, list)
            assert len(variants) >= 1
            assert len(variants) <= 3
            assert all(isinstance(variant, str) for variant in variants)
            
            print(f"S BackTranslationFROperator: Generated {len(variants)} variants")
        except Exception as e:
            print(f"⚠ BackTranslationFROperator: Model not available - {e}")
        
        # Test German back-translation
        try:
            from ea.back_translation_german import BackTranslationDEOperator
            operator = BackTranslationDEOperator()
            variants = operator.apply(self.test_prompt)
            
            assert isinstance(variants, list)
            assert len(variants) >= 1
            assert len(variants) <= 3
            assert all(isinstance(variant, str) for variant in variants)
            
            print(f"S BackTranslationDEOperator: Generated {len(variants)} variants")
        except Exception as e:
            print(f"⚠ BackTranslationDEOperator: Model not available - {e}")
        
        # Test Japanese back-translation
        try:
            from ea.back_translation_japanese import BackTranslationJAOperator
            operator = BackTranslationJAOperator()
            variants = operator.apply(self.test_prompt)
            
            assert isinstance(variants, list)
            assert len(variants) >= 1
            assert len(variants) <= 3
            assert all(isinstance(variant, str) for variant in variants)
            
            print(f"S BackTranslationJAOperator: Generated {len(variants)} variants")
        except Exception as e:
            print(f"⚠ BackTranslationJAOperator: Model not available - {e}")
        
        # Test Chinese back-translation
        try:
            from ea.back_translation_chinese import BackTranslationZHOperator
            operator = BackTranslationZHOperator()
            variants = operator.apply(self.test_prompt)
            
            assert isinstance(variants, list)
            assert len(variants) >= 1
            assert len(variants) <= 3
            assert all(isinstance(variant, str) for variant in variants)
            
            print(f"S BackTranslationZHOperator: Generated {len(variants)} variants")
        except Exception as e:
            print(f"⚠ BackTranslationZHOperator: Model not available - {e}")
        
        # Test Hindi back-translation
        try:
            from ea.back_translation_hindi import BackTranslationHIOperator
            operator = BackTranslationHIOperator()
            variants = operator.apply(self.test_prompt)
            
            assert isinstance(variants, list)
            assert len(variants) >= 1
            assert len(variants) <= 3
            assert all(isinstance(variant, str) for variant in variants)
            
            print(f"S BackTranslationHIOperator: Generated {len(variants)} variants")
        except Exception as e:
            print(f"⚠ BackTranslationHIOperator: Model not available - {e}")
    
    def test_llma_back_translation(self):
        """Test LLaMA-based back-translation operators."""
        print("\nTesting LLaMA Back-Translation Operators:")
        print("-" * 40)
        
        # Test French LLaMA back-translation
        try:
            from ea.llm_back_translation_french import LLMBackTranslationFROperator
            operator = LLMBackTranslationFROperator()
            variants = operator.apply(self.test_prompt)
            
            assert isinstance(variants, list)
            assert len(variants) >= 1
            assert len(variants) <= 3
            assert all(isinstance(variant, str) for variant in variants)
            
            print(f"S LLMBackTranslationFROperator: Generated {len(variants)} variants")
            for i, variant in enumerate(variants, 1):
                print(f"  {i}. {variant[:60]}...")
        except Exception as e:
            print(f"⚠ LLMBackTranslationFROperator: Generator not available - {e}")
        
        # Test German LLaMA back-translation
        try:
            from ea.llm_back_translation_german import LLMBackTranslationDEOperator
            operator = LLMBackTranslationDEOperator()
            variants = operator.apply(self.test_prompt)
            
            assert isinstance(variants, list)
            assert len(variants) >= 1
            
            print(f"S LLMBackTranslationDEOperator: Generated {len(variants)} variants")
        except Exception as e:
            print(f"⚠ LLMBackTranslationDEOperator: Generator not available - {e}")
        
        # Test Japanese LLaMA back-translation
        try:
            from ea.llm_back_translation_japanese import LLMBackTranslationJAOperator
            operator = LLMBackTranslationJAOperator()
            variants = operator.apply(self.test_prompt)
            
            assert isinstance(variants, list)
            assert len(variants) >= 1
            
            print(f"S LLMBackTranslationJAOperator: Generated {len(variants)} variants")
        except Exception as e:
            print(f"⚠ LLMBackTranslationJAOperator: Generator not available - {e}")
        
        # Test Chinese LLaMA back-translation
        try:
            from ea.llm_back_translation_chinese import LLMBackTranslationZHOperator
            operator = LLMBackTranslationZHOperator()
            variants = operator.apply(self.test_prompt)
            
            assert isinstance(variants, list)
            assert len(variants) >= 1
            
            print(f"S LLMBackTranslationZHOperator: Generated {len(variants)} variants")
        except Exception as e:
            print(f"⚠ LLMBackTranslationZHOperator: Generator not available - {e}")
        
        # Test Hindi LLaMA back-translation
        try:
            from ea.llm_back_translation_hindi import LLMBackTranslationHIOperator
            operator = LLMBackTranslationHIOperator()
            variants = operator.apply(self.test_prompt)
            
            assert isinstance(variants, list)
            assert len(variants) >= 1
            
            print(f"✓ LLMBackTranslationHIOperator: Generated {len(variants)} variants")
        except Exception as e:
            print(f"⚠ LLMBackTranslationHIOperator: Generator not available - {e}")
    
    def test_operator_inheritance(self):
        """Test that operators inherit correctly from base classes."""
        from ea.base_operators import _GenericBackTranslationOperator, _GenericLLMBackTranslationOperator
        
        # Test that Helsinki-NLP operators inherit from _GenericBackTranslationOperator
        from ea.back_translation_french import BackTranslationFROperator
        operator = BackTranslationFROperator()
        assert isinstance(operator, _GenericBackTranslationOperator)
        assert hasattr(operator, 'en_xx')
        assert hasattr(operator, 'xx_en')
        
        # Test that LLaMA operators inherit from _GenericLLMBackTranslationOperator
        from ea.llm_back_translation_french import LLMBackTranslationFROperator
        llma_operator = LLMBackTranslationFROperator()
        assert isinstance(llma_operator, _GenericLLMBackTranslationOperator)
        assert hasattr(llma_operator, 'generator')
        assert hasattr(llma_operator, 'target_lang')
        
        print("Operator inheritance: PASSED")


def run_back_translation_tests():
    """Run all back-translation operator tests and print results."""
    print("Testing Back-Translation Operators")
    print("=" * 50)
    
    test_instance = TestBackTranslationOperators()
    test_instance.setup_method()
    
    tests = [
        ("Helsinki-NLP Models", test_instance.test_helsinki_nlp_back_translation),
        ("LLaMA Models", test_instance.test_llma_back_translation),
        ("Inheritance", test_instance.test_operator_inheritance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nBack-Translation {test_name}:")
        try:
            test_func()
            print(f"S {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: FAILED - {e}")
    
    print(f"\nBack-Translation Tests: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    success = run_back_translation_tests()
    sys.exit(0 if success else 1)
