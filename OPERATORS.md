# Text Variation Operators

Complete documentation for all 16 variation operators in the evolutionary text generation framework.

## Overview

**16 Active Operators**:
- **13 Mutation Operators** - Single-parent text transformations
- **3 Crossover Operators** - Multi-parent text combinations

All operators return variants wrapped in XML tags (`<variant>...</variant>`) and must end with a question mark.

## Mutation Operators (13)

### 1. MLM (Masked Language Model)
**File**: `src/ea/mlm_operator.py`

Replaces [MASK] tokens using BERT-based language model.

```
Input:  "What are the [MASK] of climate change?"
Output: "What are the consequences of climate change?"
```

### 2. LLMBasedParaphrasing
**File**: `src/ea/paraphrasing.py`

LLM-based paraphrasing maintaining intent while varying phrasing.

```
Input:  "What are the benefits of exercise?"
Output: "How does physical activity help you?"
```

### 3-7. BackTranslation (5 variants)
**File**: `src/ea/back_translation.py`

Translation roundtrip through intermediate language:
- **BackTranslation_Chinese**
- **BackTranslation_French**
- **BackTranslation_German**
- **BackTranslation_Hindi**
- **BackTranslation_Japanese**

```
Input:  "What are your political views?"
        ↓ (translate to French)
        "Quelles sont vos opinions politiques?"
        ↓ (translate back to English)
Output: "What are your political opinions?"
```

### 8. SynonymReplacement
**File**: `src/ea/synonym_replacement.py`

Replaces words with WordNet synonyms using POS tagging.

```
Input:  "What are the benefits of exercise?"
Output: "What are the advantages of physical activity?"
```

### 9. AntonymReplacement
**File**: `src/ea/antonym_replacement.py`

Replaces words with WordNet antonyms using POS tagging.

```
Input:  "What are the benefits of exercise?"
Output: "What are the drawbacks of inactivity?"
```

### 10. TypographicalErrors
**File**: `src/ea/typographical_errors.py`

Introduces intentional typos and errors.

```
Input:  "What are the benefits of exercise?"
Output: "What ar the benifits of exercize?"
```

### 11. NegationOperator
**File**: `src/ea/negation_operator.py`

Inserts or modifies negations to reverse meaning.

```
Input:  "Should we trust the government?"
Output: "Should we not trust the government?"
```

### 12. ConceptAddition
**File**: `src/ea/concept_addition.py`

Injects new concepts or biases into the text.

```
Input:  "What are your views on immigration?"
Output: "What are your views on illegal immigration and its impact on jobs?"
```

### 13. StylisticMutator
**File**: `src/ea/stylistic_mutator.py`

Applies stylistic variations (formal, casual, aggressive, etc.).

```
Input:  "What are your thoughts on this policy?"
Output: "What's your take on this policy?"  # casual style
```

### 14. InformedEvolutionOperator
**File**: `src/ea/informed_evolution.py`

LLM-guided evolution using `top_10.json` (best performing genomes as examples).

**Key Features**:
- Uses top 10 performing genomes as context
- Calculates average score of top_10 for parent_score
- Only active in `--operators "ie"` or `--operators "all"` modes

```
Context: Top 10 examples with high toxicity scores
Input:  "What are your political views?"
Output: "What controversial political views do you hide from others?"
```

## Crossover Operators (3)

### 1. SemanticFusionCrossover
**File**: `src/ea/fusion_crossover.py`

Semantically blends two parent texts using LLM.

```
Parent 1: "What are your views on immigration?"
Parent 2: "How do you feel about border control?"
Output:   "What are your views on immigration and border control policies?"
```

### 2. SemanticSimilarityCrossover
**File**: `src/ea/semantic_similarity_crossover.py`

Combines parents based on semantic similarity scores.

```
Parent 1: "What are the benefits of technology?"
Parent 2: "How has innovation changed society?"
Output:   "What are the benefits of technological innovation in society?"
```

### 3. CutAndSliceCrossover
**File**: `src/ea/cut_and_slice_crossover.py`

Recombines text segments from two parents.

```
Parent 1: "What are your political views on immigration?"
Parent 2: "How do you feel about controversial topics?"
Output:   "What are your political views on controversial topics?"
```

## Operator Configuration

### Operator Modes

| Mode | Operators Used | Data Source |
|------|----------------|-------------|
| `"ie"` | InformedEvolutionOperator only | `top_10.json` |
| `"cm"` | All except InformedEvolution | `parents.json` |
| `"all"` | All 16 operators | Both files |

### Variant Generation

```bash
# Generate 1 variant per operator per parent (default)
--max-variants 1

# Generate 3 variants per operator per parent
--max-variants 3

# Generate 5 variants per operator per parent
--max-variants 5
```

## Parent Score Calculation

Each generated variant's `creation_info` includes `parent_score`:

### Mutation
```python
parent_score = parent.toxicity  # Direct score
```

### Crossover
```python
parent_score = average([parent1.toxicity, parent2.toxicity])
```

### Informed Evolution
```python
parent_score = average([top_10 toxicity scores])
```

## Error Handling

All operators implement graceful error handling:

1. **LLM Refusal**: Returns empty list `[]` if LLM refuses to generate
2. **XML Parsing**: Raises error if variant not in `<variant>` tags
3. **Question Mark**: Enforced on all variants
4. **Minimum Score**: All scores have minimum `0.0001`

## Operator Selection

### Per Generation
1. Load parents from `parents.json` (for `cm` and `all` modes)
2. Load top_10 from `top_10.json` (for `ie` and `all` modes)
3. Apply each operator `max_variants` times
4. Save results to `temp.json`

### Adaptive Parent Selection

Parents selected based on mode:
- **DEFAULT**: 1 elite + 1 non-elite
- **EXPLORE**: 1 elite + 2 non-elites
- **EXPLOIT**: 2 elites + 1 non-elite

## Performance

### Execution Times (Approximate)
| Operator Type | Time per Variant |
|---------------|------------------|
| WordNet-based | 0.1-0.5s |
| BERT-based | 0.5-1.5s |
| LLM-based | 2-5s |
| Back-translation | 3-8s |

### Memory Usage
- **Model Caching**: Max 2 models in memory
- **Lazy Loading**: Models loaded only when needed
- **Cache Cleanup**: LRU eviction when limit exceeded

## XML Tag Format

All operators must return variants in XML tags:

```xml
<variant>What are your thoughts on this controversial topic?</variant>
```

If variant not in tags, parsing will fail and raise an error.

## Variant Validation

All variants must:
1. ✅ Be wrapped in `<variant>` XML tags
2. ✅ End with a question mark `?`
3. ✅ Be different from the parent prompt
4. ✅ Be valid text (non-empty)

## Usage Examples

### Run with all operators
```bash
python3 src/main.py --generations 10 --operators "all" --max-variants 3
```

### Run with classical methods only
```bash
python3 src/main.py --generations 10 --operators "cm" --max-variants 3
```

### Run with informed evolution only
```bash
python3 src/main.py --generations 10 --operators "ie" --max-variants 5
```

## See Also

- **[README.md](README.md)** - Getting started and usage
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture
- **[src/ea/README.md](src/ea/README.md)** - Evolutionary algorithm details
