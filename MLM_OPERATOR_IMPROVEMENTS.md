# MLM Operator Improvements Summary

## Overview
The MLM (Masked Language Model) operator has been significantly enhanced to address the core issue where `<MASKED>` tokens were not being properly replaced by the LLM. The improvements implement a robust 3-step process with structured output parsing and comprehensive error handling.

## Key Issues Identified from Log Analysis

### 1. **LLM Responses Contained `<MASKED>` Tokens**
- **Problem**: The LLM was not properly replacing mask tokens, often returning text that still contained `<MASKED>` tokens
- **Evidence**: Log entries showed responses like "is a well-known example of a `<MASKED>` sentence..." and "The `<MASKED>` of the `<MASKED>` was a `<MASKED>` man..."

### 2. **Poor Prompt Design**
- **Problem**: The original prompt was too simple and didn't enforce structured output
- **Evidence**: LLM responses were often meta-textual explanations rather than direct replacements

### 3. **No Response Parsing**
- **Problem**: The operator didn't parse or validate LLM responses
- **Evidence**: Even when LLM produced valid completions, they weren't properly extracted and applied

### 4. **Parameter Validation Issues**
- **Problem**: Invalid `max_variants` parameters caused repeated warnings
- **Evidence**: Multiple log entries showing "MLM: invalid max_variants, defaulting to 3"

## Implemented Solutions

### 1. **Enhanced 3-Step Process**

#### Step 1: Numbered Masking
- **Before**: Used generic `<MASKED>` tokens
- **After**: Uses numbered tokens `<MASKED_1>`, `<MASKED_2>`, etc.
- **Benefits**: 
  - Better tracking of individual masks
  - Enables structured replacement mapping
  - Prevents confusion between multiple masks

#### Step 2: Structured LLM Prompting
- **Before**: Simple prompt asking for completion
- **After**: Structured JSON-based prompt with clear requirements
- **Benefits**:
  - Forces LLM to return structured output
  - Includes both individual replacements and completed text
  - Clear validation criteria

#### Step 3: Robust Response Parsing & Validation
- **Before**: No parsing, direct use of LLM output
- **After**: Multi-level parsing with fallback strategies
- **Benefits**:
  - Extracts structured data from LLM responses
  - Handles markdown code blocks and formatting issues
  - Validates mask count matches expectations
  - Fallback to simpler completion if structured parsing fails

### 2. **Improved Parameter Validation**

```python
def _validate_max_variants(self, max_variants: Any) -> int:
    """Validate and convert max_variants to positive integer."""
    try:
        if isinstance(max_variants, str):
            max_variants = int(max_variants)
        elif isinstance(max_variants, float):
            max_variants = int(max_variants)
        
        val = max(1, int(max_variants))
        if val < 1:
            self.logger.warning(f"{self.name}: max_variants < 1, setting to 1")
            return 1
        return val
    except (ValueError, TypeError) as e:
        self.logger.warning(f"{self.name}: Invalid max_variants '{max_variants}', using default 3: {e}")
        return 3
```

### 3. **Structured Prompt Design**

The new prompt enforces structured output:

```
You are a text completion expert. I need you to replace numbered mask tokens with appropriate words.

TASK: Replace each <MASKED_N> token in the text with a single word that makes sense in context.

TEXT TO COMPLETE:
"The <MASKED_1> brown fox <MASKED_2> over the lazy dog."

REQUIREMENTS:
1. Replace exactly 2 mask tokens: ['<MASKED_1>', '<MASKED_2>']
2. Each replacement should be a single word
3. The completed text should be grammatically correct and semantically coherent
4. Return ONLY a JSON object with this exact format:

{
  "replacements": {
    "1": "word_for_MASKED_1",
    "2": "word_for_MASKED_2"
  },
  "completed_text": "The final text with all masks replaced"
}

IMPORTANT: Return ONLY the JSON object, no additional text or explanations.
```

### 4. **Robust Response Parsing**

```python
def _parse_llm_response(self, response: str, mask_mapping: Dict[int, str]) -> Optional[Dict[str, Any]]:
    """Parse LLM response to extract replacements and completed text."""
    try:
        # Clean response
        cleaned_response = response.strip()
        
        # Remove markdown code blocks if present
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        while cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        # Try to parse JSON
        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                return None
        
        # Validate structure and mask count
        # ... validation logic ...
        
        return {
            "replacements": replacements,
            "completed_text": completed_text.strip()
        }
        
    except Exception as e:
        self.logger.debug(f"{self.name}: Failed to parse LLM response: {e}")
        return None
```

### 5. **Fallback Mechanisms**

When structured parsing fails, the operator falls back to simpler completion:

```python
def _fallback_replacement(self, masked_text: str, mask_mapping: Dict[int, str]) -> str:
    """Fallback replacement strategy when structured parsing fails."""
    try:
        # Simple fallback: ask for direct completion
        fallback_prompt = f"""Complete this text by replacing each <MASKED_N> token with an appropriate word. Return only the completed text, no explanations.

Text: "{masked_text}"

Completed text:"""

        response = self.generator.generate_response(fallback_prompt)
        if response:
            completed = response.strip().strip('"').strip("'")
            # Basic validation
            if "<MASKED_" not in completed and len(completed) > 0:
                self.logger.info(f"{self.name}: Fallback replacement successful")
                return completed
        
        # Last resort: return original masked text
        self.logger.warning(f"{self.name}: All replacement strategies failed")
        return masked_text
        
    except Exception as e:
        self.logger.error(f"{self.name}: Fallback replacement failed: {e}")
        return masked_text
```

### 6. **Enhanced Logging and Debugging**

- **Step-by-step logging**: Each step of the 3-step process is logged with clear indicators
- **Mask mapping tracking**: Shows which original words were masked and their numbers
- **Replacement validation**: Logs successful replacements and validation failures
- **Error context**: Provides detailed error information for debugging

### 7. **Comprehensive Test Suite Updates**

Added new tests specifically for the enhanced functionality:

- **`test_structured_llm_response_parsing`**: Tests JSON parsing with various formats
- **`test_llm_fill_structured_basic`**: Tests the new structured fill process
- **Updated masking tests**: Now test numbered masks and mask mapping

## Expected Improvements

### 1. **Higher Success Rate**
- **Before**: ~0% success rate (based on log analysis)
- **After**: Expected 80-90% success rate with structured prompting and fallbacks

### 2. **Better Error Handling**
- **Before**: Silent failures or unclear error states
- **After**: Clear error messages, fallback strategies, and graceful degradation

### 3. **Improved Reliability**
- **Before**: Inconsistent behavior with different inputs
- **After**: Robust parameter validation and consistent processing

### 4. **Better Debugging**
- **Before**: Limited logging made issues hard to diagnose
- **After**: Comprehensive logging at each step with clear success/failure indicators

## Usage Example

```python
from ea.mlm_operator import MLMOperator

# Initialize with improved parameter handling
operator = MLMOperator(max_variants=3, seed=42)

# Apply the 3-step process
result = operator.apply("The quick brown fox jumps over the lazy dog.")
# Expected: ["The fast brown fox leaps over the lazy dog."] or similar variant

# The process now:
# 1. Masks words: "The <MASKED_1> brown fox <MASKED_2> over the lazy dog."
# 2. Gets structured LLM response with replacements
# 3. Validates and applies replacements
# 4. Returns completed variant
```

## Migration Notes

- **Backward Compatibility**: The `apply()` method signature remains the same
- **New Dependencies**: Added `json` and `re` imports for parsing
- **Logging Changes**: More verbose logging may require log level adjustments
- **Performance**: Slightly increased processing time due to structured parsing, but much higher success rate

## Testing Recommendations

1. **Run the updated test suite**: `python tests/test_mlm_operator.py`
2. **Monitor logs**: Check for successful mask replacements in log output
3. **Validate outputs**: Ensure returned text doesn't contain `<MASKED_>` tokens
4. **Test edge cases**: Empty text, single words, very long text
5. **Performance testing**: Verify acceptable processing times with your LLM setup

## Future Enhancements

1. **Caching**: Cache successful replacement patterns for reuse
2. **Multiple LLM backends**: Support for different LLM providers
3. **Confidence scoring**: Rate replacement quality
4. **Batch processing**: Process multiple texts simultaneously
5. **Custom validation**: User-defined validation rules for replacements
