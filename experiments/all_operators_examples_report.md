# All Operators Variant Creation Report

## Summary

**Total Examples Found**: 47 generations across 9 out of 10 comb runs where **ALL 12 operators** created at least 1 variant.

## Key Statistics

- **Total comb runs checked**: 10
- **Runs with examples**: 9 (90%)
- **Total examples**: 47 generations
- **Average variants per generation**: 22.6
- **Min variants**: 16
- **Max variants**: 31

## Examples by Run

### run01_comb: 6 generations
- Generation 13: 16 variants (14 mutation, 2 crossover)
- Generation 32: 30 variants (26 mutation, 4 crossover)
- Generation 41: 21 variants (19 mutation, 2 crossover)
- Generation 48: 23 variants (18 mutation, 5 crossover)
- Generation 49: 24 variants (22 mutation, 2 crossover)
- Generation 50: 27 variants (21 mutation, 6 crossover)

### run02_comb: 4 generations
- Generation 22: 24 variants (19 mutation, 5 crossover)
- Generation 38: 21 variants (17 mutation, 4 crossover)
- Generation 40: 25 variants (22 mutation, 3 crossover)
- Generation 47: 21 variants (19 mutation, 2 crossover)

### run03_comb: 9 generations
- Generation 8: 23 variants (20 mutation, 3 crossover)
- Generation 10: 27 variants (23 mutation, 4 crossover)
- Generation 25: 21 variants (18 mutation, 3 crossover)
- Generation 31: 20 variants (18 mutation, 2 crossover)
- Generation 32: 17 variants (15 mutation, 2 crossover)
- Generation 33: 19 variants (16 mutation, 3 crossover)
- Generation 36: 18 variants (15 mutation, 3 crossover)
- Generation 42: 20 variants (18 mutation, 2 crossover)
- Generation 43: 21 variants (19 mutation, 2 crossover)

### run04_comb: 7 generations
- Generation 10: 24 variants (22 mutation, 2 crossover)
- Generation 13: 22 variants (19 mutation, 3 crossover)
- Generation 19: 23 variants (21 mutation, 2 crossover)
- Generation 20: 25 variants (22 mutation, 3 crossover)
- Generation 21: 23 variants (19 mutation, 4 crossover)
- Generation 38: 22 variants (18 mutation, 4 crossover)
- Generation 45: 26 variants (23 mutation, 3 crossover)

### run05_comb: 4 generations
- Generation 8: 25 variants (22 mutation, 3 crossover)
- Generation 38: 22 variants (19 mutation, 3 crossover)
- Generation 42: 23 variants (20 mutation, 3 crossover)
- Generation 45: 18 variants (15 mutation, 3 crossover)

### run06_comb: 11 generations (Most examples)
- Generation 26: 18 variants (16 mutation, 2 crossover)
- Generation 28: 28 variants (23 mutation, 5 crossover)
- Generation 29: 31 variants (25 mutation, 6 crossover) ⭐ **Highest variant count**
- Generation 31: 19 variants (17 mutation, 2 crossover)
- Generation 33: 28 variants (24 mutation, 4 crossover)
- Generation 34: 24 variants (20 mutation, 4 crossover)
- Generation 38: 27 variants (25 mutation, 2 crossover)
- Generation 42: 24 variants (21 mutation, 3 crossover)
- Generation 43: 20 variants (18 mutation, 2 crossover)
- Generation 46: 24 variants (21 mutation, 3 crossover)
- Generation 48: 25 variants (21 mutation, 4 crossover)

### run08_comb: 4 generations
- Generation 40: 22 variants (19 mutation, 3 crossover)
- Generation 41: 23 variants (19 mutation, 4 crossover)
- Generation 46: 24 variants (21 mutation, 3 crossover)
- Generation 50: 18 variants (15 mutation, 3 crossover)

### run09_comb: 1 generation
- Generation 31: 19 variants (17 mutation, 2 crossover)

### run10_comb: 1 generation
- Generation 7: 18 variants (16 mutation, 2 crossover)

## Best Examples

### Highest Variant Count
- **run06_comb, Generation 29**: 31 variants (25 mutation, 6 crossover)
  - All 12 operators created variants
  - Highest total variant count across all examples

### Earliest Generation
- **run10_comb, Generation 7**: 18 variants (16 mutation, 2 crossover)
  - Earliest generation where all operators succeeded

### Most Consistent Run
- **run06_comb**: 11 generations with all operators
  - Most examples in a single run

## Operators (All 12)

1. ConceptAdditionOperator
2. InformedEvolutionOperator
3. LLMBackTranslation_HI
4. LLMBasedParaphrasing
5. LLM_POSAwareSynonymReplacement
6. MLM
7. NegationOperator
8. POSAwareAntonymReplacement
9. SemanticFusionCrossover
10. SemanticSimilarityCrossover
11. StylisticMutator
12. TypographicalErrorsOperator

## Conclusion

The analysis confirms that in **comb mode (all operators)**, parents can successfully create variants using all 12 operators in a single generation. This occurs frequently across multiple runs, with 47 examples found across 9 out of 10 comb runs.

The fact that all operators can create variants demonstrates:
- Effective parent selection
- Successful operator execution
- Diverse variant generation
- Robust evolution process

