# Variation Operators: Scientific Analysis

## Overview

This document provides a comprehensive analysis of the 16 variation operators used in the evolutionary optimization framework for AI safety testing.

## Operator Classification

### Mutation Operators (13)
Single-parent operators that modify existing prompts through various linguistic transformations.

### Crossover Operators (3)
Multi-parent operators that combine genetic material from multiple prompts to create offspring.

## Detailed Operator Analysis

### 1. Informed Evolution Operator

**Type**: Mutation  
**Scientific Basis**: LLM-guided evolution using top performers as examples

**Algorithm**:
```
1. Load top 10 highest-scoring prompts
2. Use LLM to generate evolved variants based on examples
3. Apply structured output parsing
4. Validate question format and length
```

**Mathematical Properties**:
- Parent score: Average of top 10 examples
- Success rate: ~60-80% (depends on LLM quality)
- Diversity: High (LLM creativity)

### 2. Masked Language Model (MLM) Operator

**Type**: Mutation  
**Scientific Basis**: Contextual word substitution using transformer models

**Algorithm**:
```
1. Mask random words in prompt
2. Use MLM to predict replacements
3. Generate multiple variants
4. Filter for question format
```

**Mathematical Properties**:
- Parent score: Single parent toxicity
- Success rate: ~40-60%
- Diversity: Medium (contextual constraints)

### 3. Paraphrasing Operator

**Type**: Mutation  
**Scientific Basis**: Semantic-preserving text transformation

**Algorithm**:
```
1. Use LLM to paraphrase prompt
2. Maintain semantic meaning
3. Apply structured output parsing
4. Validate question format
```

**Mathematical Properties**:
- Parent score: Single parent toxicity
- Success rate: ~70-90%
- Diversity: Medium (semantic constraints)

### 4. Back Translation Operators (5 variants)

**Type**: Mutation  
**Scientific Basis**: Multi-language roundtrip translation

**Languages**: Chinese, French, German, Hindi, Japanese

**Algorithm**:
```
1. Translate prompt to target language
2. Translate back to English
3. Generate multiple variants
4. Filter for question format
```

**Mathematical Properties**:
- Parent score: Single parent toxicity
- Success rate: ~30-50% (translation quality dependent)
- Diversity: High (translation artifacts)

### 5. Synonym/Antonym Replacement

**Type**: Mutation  
**Scientific Basis**: Lexical substitution with POS awareness

**Algorithm**:
```
1. Identify POS tags in prompt
2. Find synonyms/antonyms for each word
3. Replace with POS-appropriate alternatives
4. Generate multiple variants
```

**Mathematical Properties**:
- Parent score: Single parent toxicity
- Success rate: ~50-70%
- Diversity: Medium (lexical constraints)

### 6. Negation Operator

**Type**: Mutation  
**Scientific Basis**: Logical operator insertion

**Algorithm**:
```
1. Identify logical operators in prompt
2. Insert negation operators
3. Generate multiple variants
4. Filter for question format
```

**Mathematical Properties**:
- Parent score: Single parent toxicity
- Success rate: ~40-60%
- Diversity: Low (logical constraints)

### 7. Concept Addition Operator

**Type**: Mutation  
**Scientific Basis**: Semantic concept injection

**Algorithm**:
```
1. Identify semantic concepts in prompt
2. Inject related concepts
3. Generate multiple variants
4. Filter for question format
```

**Mathematical Properties**:
- Parent score: Single parent toxicity
- Success rate: ~60-80%
- Diversity: High (concept diversity)

### 8. Typographical Errors Operator

**Type**: Mutation  
**Scientific Basis**: Character-level noise injection

**Algorithm**:
```
1. Identify character positions in prompt
2. Inject typographical errors
3. Generate multiple variants
4. Filter for question format
```

**Mathematical Properties**:
- Parent score: Single parent toxicity
- Success rate: ~30-50%
- Diversity: Medium (error patterns)

### 9. Stylistic Mutator

**Type**: Mutation  
**Scientific Basis**: Writing style transformation

**Algorithm**:
```
1. Identify writing style elements
2. Transform style characteristics
3. Generate multiple variants
4. Filter for question format
```

**Mathematical Properties**:
- Parent score: Single parent toxicity
- Success rate: ~50-70%
- Diversity: High (style diversity)

### 10. Semantic Similarity Crossover

**Type**: Crossover  
**Scientific Basis**: Crossbreeding based on semantic distance

**Algorithm**:
```
1. Calculate semantic similarity between parents
2. Select similar parents for crossover
3. Combine genetic material
4. Generate multiple variants
```

**Mathematical Properties**:
- Parent score: Average of parent scores
- Success rate: ~60-80%
- Diversity: Medium (similarity constraints)

### 11. Semantic Fusion Crossover

**Type**: Crossover  
**Scientific Basis**: Hybrid prompt generation

**Algorithm**:
```
1. Identify semantic elements in parents
2. Fuse elements from multiple parents
3. Generate hybrid variants
4. Filter for question format
```

**Mathematical Properties**:
- Parent score: Average of parent scores
- Success rate: ~70-90%
- Diversity: High (fusion creativity)

### 12. Cut-and-Slice Crossover

**Type**: Crossover  
**Scientific Basis**: Structural recombination

**Algorithm**:
```
1. Identify structural elements in parents
2. Cut and slice structural components
3. Recombine structural elements
4. Generate multiple variants
```

**Mathematical Properties**:
- Parent score: Average of parent scores
- Success rate: ~50-70%
- Diversity: Medium (structural constraints)

## Performance Metrics

### Success Rates by Operator Type
| Operator Type | Success Rate | Diversity | Computational Cost |
|---------------|--------------|-----------|-------------------|
| Informed Evolution | 60-80% | High | High |
| MLM | 40-60% | Medium | Medium |
| Paraphrasing | 70-90% | Medium | High |
| Back Translation | 30-50% | High | Medium |
| Synonym/Antonym | 50-70% | Medium | Low |
| Negation | 40-60% | Low | Low |
| Concept Addition | 60-80% | High | Medium |
| Typographical Errors | 30-50% | Medium | Low |
| Stylistic Mutator | 50-70% | High | Medium |
| Semantic Similarity | 60-80% | Medium | Medium |
| Semantic Fusion | 70-90% | High | High |
| Cut-and-Slice | 50-70% | Medium | Low |

### Convergence Analysis

**Fast Convergence** (5-15 generations):
- Informed Evolution
- Semantic Fusion
- Paraphrasing

**Medium Convergence** (15-30 generations):
- MLM
- Concept Addition
- Semantic Similarity

**Slow Convergence** (30+ generations):
- Back Translation
- Typographical Errors
- Negation