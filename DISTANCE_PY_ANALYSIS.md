# Distance.py Analysis: Inconsistencies Found and Fixed

## Summary

Analyzed `src/speciation/distance.py` and found **3 inconsistencies** that have been fixed.

---

## Issues Found and Fixed

### ✅ Issue 1: Incorrect Documentation in `semantic_distance()`

**Problem**:
- Docstring incorrectly stated that `semantic_distance()` is used for `theta_sim` and `theta_merge` thresholds
- Actually, these thresholds are used with `ensemble_distance()`, not `semantic_distance()`
- `semantic_distance()` is only used internally by `ensemble_distance()` to compute the genotype component

**Root Cause**: Outdated documentation that didn't reflect the current implementation using ensemble distance.

**Fix Applied**:
- Updated docstring to clarify that `semantic_distance()` is used internally by `ensemble_distance()`
- Removed incorrect references to theta_sim/theta_merge thresholds

**File Modified**: `src/speciation/distance.py` (lines 29-33)

---

### ✅ Issue 2: Type Handling Inconsistency in `ensemble_distances_batch()`

**Problem**:
- Function accepts `phenotypes: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]]`
- Code at line 183 did `for i, p in enumerate(phenotypes)` which works for both types
- BUT: If `phenotypes` is a numpy array, it can't contain None values (numpy arrays are homogeneous)
- The code didn't distinguish between numpy array and list cases
- This could cause incorrect behavior when phenotypes is a numpy array

**Root Cause**: Type hint allows both numpy array and list, but code didn't handle them differently.

**Fix Applied**:
- Added explicit type checking: `isinstance(phenotypes, np.ndarray)` vs list
- For numpy arrays: Handle directly (no None values possible)
- For lists: Handle None values as before
- Ensures correct behavior for both input types

**File Modified**: `src/speciation/distance.py` (lines 179-207)

---

### ✅ Issue 3: Length Calculation Bug in `ensemble_distances_batch()`

**Problem**:
- When `embeddings` is 1D, `len(embeddings)` returns the embedding dimension, not the number of targets
- Line 206 used `len(embeddings)` to create phenotype distance array
- This could create arrays with wrong size if embeddings is 1D
- `semantic_distances_batch()` handles 1D->2D conversion, but we need `num_targets` before that

**Root Cause**: Inconsistent handling of 1D vs 2D embeddings for length calculation.

**Fix Applied**:
- Calculate `num_targets` explicitly before calling `semantic_distances_batch()`
- Handle 1D embeddings by reshaping and setting `num_targets = 1`
- Use `num_targets` consistently for all array creation (not `len(embeddings)` or `len(phenotypes)`)
- Ensures correct array sizes regardless of input shape

**File Modified**: `src/speciation/distance.py` (lines 171-207)

---

## Code Changes Summary

### Before:
```python
# Compute genotype distances (range [0, 2])
d_genotype = semantic_distances_batch(query_embedding, embeddings)

# ... phenotype handling with len(phenotypes) or len(embeddings) ...
```

### After:
```python
# Ensure embeddings is 2D for consistent length calculation
if embeddings.ndim == 1:
    num_targets = 1
    embeddings_2d = embeddings.reshape(1, -1)
else:
    num_targets = len(embeddings)
    embeddings_2d = embeddings

# Compute genotype distances (range [0, 2])
d_genotype = semantic_distances_batch(query_embedding, embeddings_2d)

# ... phenotype handling with explicit type checking and num_targets ...
```

---

## Validation

- ✅ All syntax checks pass
- ✅ No linter errors
- ✅ Type handling now consistent for both numpy arrays and lists
- ✅ Length calculations now correct for 1D and 2D embeddings
- ✅ Documentation now accurate

---

## Impact

These fixes ensure:
1. **Correct behavior** when `phenotypes` is a numpy array (no None values)
2. **Correct behavior** when `phenotypes` is a list with None values
3. **Correct array sizing** regardless of input embedding shape (1D or 2D)
4. **Accurate documentation** that reflects actual usage

---

**Date**: 2026-01-13  
**File**: `src/speciation/distance.py`  
**Status**: ✅ All inconsistencies fixed
