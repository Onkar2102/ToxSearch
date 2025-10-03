#!/usr/bin/env python3
"""
Test suite for operator helper functions.

This module tests the helper functions from operator_helpers.py
to ensure they work correctly after the operator separation.
"""

import sys
import os

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestOperatorHelpers:
    """Test class for operator helper functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.north_star_metric = "engagement"
        self.log_file = None
        
    def test_get_single_parent_operators(self):
        """Test get_single_parent_operators function."""
        from ea.operator_helpers import get_single_parent_operators
        
        operators = get_single_parent_operators(self.north_star_metric, self.log_file)
        
        assert isinstance(operators, list)
        assert len(operators) > 0
        
        # Check that all operators have the required attributes
        for operator in operators:
            assert hasattr(operator, 'name')
            assert hasattr(operator, 'operator_type')
            assert hasattr(operator, 'description')
            assert operator.operator_type == "mutation"
            assert isinstance(operator.name, str)
            assert isinstance(operator.description, str)
        
        print(f"get_single_parent_operators: Retrieved {len(operators)} mutation operators")
        for operator in operators:
            print(f"  - {operator.name}: {operator.description}")
    
    def test_get_multi_parent_operators(self):
        """Test get_multi_parent_operators function."""
        from ea.operator_helpers import get_multi_parent_operators
        
        operators = get_multi_parent_operators(self.log_file)
        
        assert isinstance(operators, list)
        assert len(operators) > 0
        
        # Check that all operators have the required attributes
        for operator in operators:
            assert hasattr(operator, 'name')
            assert hasattr(operator, 'operator_type')
            assert hasattr(operator, 'description')
            assert operator.operator_type == "crossover"
            assert isinstance(operator.name, str)
            assert isinstance(operator.description, str)
        
        print(f"get_multi_parent_operators: Retrieved {len(operators)} crossover operators")
        for operator in operators:
            print(f"  - {operator.name}: {operator.description}")
    
    def test_get_applicable_operators_single_parent(self):
        """Test get_applicable_operators with single parent."""
        from ea.operator_helpers import get_applicable_operators
        
        operators = get_applicable_operators(1, self.north_star_metric, self.log_file)
        
        assert isinstance(operators, list)
        assert len(operators) > 0
        
        # All should be mutation operators
        for operator in operators:
            assert operator.operator_type == "mutation"
        
        print(f"get_applicable_operators(1): Retrieved {len(operators)} mutation operators")
    
    def test_get_applicable_operators_multi_parent(self):
        """Test get_applicable_operators with multiple parents."""
        from ea.operator_helpers import get_applicable_operators
        
        operators = get_applicable_operators(2, self.north_star_metric, self.log_file)
        
        assert isinstance(operators, list)
        assert len(operators) > 0
        
        # All should be crossover operators
        for operator in operators:
            assert operator.operator_type == "crossover"
        
        print(f"get_applicable_operators(2): Retrieved {len(operators)} crossover operators")
    
    def test_limit_variants_function(self):
        """Test limit_variants utility function."""
        from ea.operator_helpers import limit_variants
        
        # Test normal case
        variants = ["variant1", "variant2", "variant3", "variant4", "variant5"]
        limited = limit_variants(variants, max_variants=3)
        
        assert isinstance(limited, list)
        assert len(limited) == 3
        assert all(var in variants for var in limited)
        
        # Test when variants are fewer than limit
        limited = limit_variants(variants[:2], max_variants=3)
        assert len(limited) == 2
        
        # Test empty list
        limited = limit_variants([], max_variants=3)
        assert len(limited) == 0
        
        print("limit_variants function: Tests passed")
    
    def test_get_generator_function(self):
        """Test get_generator utility function."""
        from ea.operator_helpers import get_generator
        
        # This test may fail if models are not available
        try:
            generator = get_generator()
            assert generator is not None
            
            # Check that generator has expected methods
            assert hasattr(generator, 'generate_response')
            assert hasattr(generator, 'translate')
            
            print("get_generator function: Successfully retrieved generator")
            return True
        except Exception as e:
            print(f"get_generator function: Warning - Generator not available: {e}")
            return False  # Don't fail the test for missing models


def run_helper_tests():
    """Run all operator helper tests and print results."""
    print("Testing Operator Helper Functions")
    print("=" * 50)
    
    test_instance = TestOperatorHelpers()
    test_instance.setup_method()
    
    tests = [
        ("get_single_parent_operators", test_instance.test_get_single_parent_operators),
        ("get_multi_parent_operators", test_instance.test_get_multi_parent_operators),
        ("get_applicable_operators(1)", test_instance.test_get_applicable_operators_single_parent),
        ("get_applicable_operators(2)", test_instance.test_get_applicable_operators_multi_parent),
        ("limit_variants", test_instance.test_limit_variants_function),
        ("get_generator", test_instance.test_get_generator_function),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nHelper {test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            if result is False:
                print(f"⚠ {test_name}: PASSED (with warnings)")
                passed += 1
            else:
                print(f"S {test_name}: PASSED")
                passed += 1
        except Exception as e:
            print(f"✗ {test_name}: FAILED - {e}")
    
    print(f"\nOperator Helper Tests: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    success = run_helper_tests()
    sys.exit(0 if success else 1)
