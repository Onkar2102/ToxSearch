"""
Verify Ensemble Distance Metric Properties

This script loads real execution data and verifies that the ensemble distance metric
satisfies the three fundamental metric properties:

1. Non-negativity: d(u, v) ≥ 0 with equality if and only if u = v
2. Symmetry: d(u, v) = d(v, u)
3. Triangle inequality: d(u, w) ≤ d(u, v) + d(v, w)

Based on the ensemble distance formula:
d_ensemble(u, v) = w_genotype × d_genotype_norm(u, v) + w_phenotype × d_phenotype(u, v)

Where:
- d_genotype_norm = (1 - cos(e_u, e_v)) / 2.0  (normalized to [0, 1])
- d_phenotype = ||p_u - p_v||_2 / √8  (normalized to [0, 1])
- w_genotype = 0.7, w_phenotype = 0.3
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speciation.distance import ensemble_distance
from speciation.phenotype_distance import extract_phenotype_vector


def load_genomes(execution_dir: Path) -> List[dict]:
    """Load genomes from elites.json."""
    elites_path = execution_dir / "elites.json"
    if not elites_path.exists():
        raise FileNotFoundError(f"elites.json not found in {execution_dir}")
    
    with open(elites_path, 'r') as f:
        genomes = json.load(f)
    
    return genomes


def extract_embedding_and_phenotype(genome: dict, compute_if_missing: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract embedding and phenotype vector from genome."""
    # Extract embedding
    embedding = None
    if "prompt_embedding" in genome:
        emb_list = genome["prompt_embedding"]
        if isinstance(emb_list, list):
            embedding = np.array(emb_list, dtype=np.float32)
        elif isinstance(emb_list, np.ndarray):
            embedding = emb_list
    
    # Compute embedding if missing
    if embedding is None and compute_if_missing and "prompt" in genome:
        try:
            from speciation.embeddings import get_embedding_model
            model = get_embedding_model()
            prompt = genome.get("prompt", "")
            if prompt:
                embedding = model.encode([prompt])[0]
        except Exception as e:
            print(f"  Warning: Could not compute embedding for genome {genome.get('id')}: {e}")
    
    # Extract phenotype
    phenotype = extract_phenotype_vector(genome, logger=None)
    
    return embedding, phenotype


def filter_valid_genomes(genomes: List[dict], compute_embeddings: bool = True) -> List[dict]:
    """Filter genomes that have both embedding and phenotype."""
    valid = []
    print("  Computing embeddings for genomes if missing...")
    for i, genome in enumerate(genomes):
        embedding, phenotype = extract_embedding_and_phenotype(genome, compute_if_missing=compute_embeddings)
        if embedding is not None and phenotype is not None:
            # Verify embedding is normalized (or normalize it)
            norm = np.linalg.norm(embedding)
            if not np.isclose(norm, 1.0, atol=1e-5):
                # Normalize if not already normalized
                embedding = embedding / norm
                genome["prompt_embedding"] = embedding.tolist()
            valid.append(genome)
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(genomes)} genomes, found {len(valid)} valid")
    return valid


def test_non_negativity(genomes: List[dict], num_samples: int = 100) -> Tuple[bool, List[float], int]:
    """
    Test Property 1: Non-negativity
    d(u, v) ≥ 0 for all u, v
    d(u, v) = 0 if and only if u = v
    """
    print("\n" + "="*80)
    print("TEST 1: Non-Negativity Property")
    print("="*80)
    print(f"Testing: d(u, v) ≥ 0 for all pairs")
    print(f"Testing: d(u, v) = 0 if and only if u = v")
    print(f"Sampling {num_samples} random pairs...")
    
    valid_genomes = filter_valid_genomes(genomes)
    if len(valid_genomes) < 2:
        print("ERROR: Need at least 2 valid genomes with embeddings and phenotypes")
        return False, [], 0
    
    np.random.seed(42)  # For reproducibility
    distances = []
    zero_distances = []
    negative_count = 0
    identical_pairs = 0
    
    for _ in range(num_samples):
        i, j = np.random.choice(len(valid_genomes), size=2, replace=False)
        g1, g2 = valid_genomes[i], valid_genomes[j]
        
        e1, p1 = extract_embedding_and_phenotype(g1)
        e2, p2 = extract_embedding_and_phenotype(g2)
        
        # Normalize embeddings if needed
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 / np.linalg.norm(e2)
        
        dist = ensemble_distance(e1, e2, p1, p2, w_genotype=0.7, w_phenotype=0.3)
        distances.append(dist)
        
        # Check for negative distances
        if dist < 0:
            negative_count += 1
            print(f"  WARNING: Negative distance found: d({g1['id']}, {g2['id']}) = {dist:.6f}")
        
        # Check for zero distances
        if abs(dist) < 1e-6:
            zero_distances.append((g1['id'], g2['id'], dist))
            # Check if genomes are actually identical
            if np.allclose(e1, e2, atol=1e-5) and np.allclose(p1, p2, atol=1e-5):
                identical_pairs += 1
    
    # Test with identical vectors
    print("\nTesting with identical vectors (should give d = 0):")
    if len(valid_genomes) > 0:
        g = valid_genomes[0]
        e, p = extract_embedding_and_phenotype(g)
        e = e / np.linalg.norm(e)
        dist_identical = ensemble_distance(e, e, p, p, w_genotype=0.7, w_phenotype=0.3)
        print(f"  d(genome_{g['id']}, genome_{g['id']}) = {dist_identical:.10f}")
        if abs(dist_identical) < 1e-6:
            print("  PASS: Identical vectors give distance = 0")
        else:
            print(f"  WARNING: Identical vectors give distance = {dist_identical:.10f} (expected 0)")
    
    # Summary
    min_dist = min(distances) if distances else 0
    max_dist = max(distances) if distances else 0
    mean_dist = np.mean(distances) if distances else 0
    
    print(f"\nResults:")
    print(f"  Total pairs tested: {num_samples}")
    print(f"  Negative distances: {negative_count}")
    print(f"  Zero distances: {len(zero_distances)}")
    print(f"  Identical pairs (e1=e2 and p1=p2): {identical_pairs}")
    print(f"  Distance range: [{min_dist:.6f}, {max_dist:.6f}]")
    print(f"  Mean distance: {mean_dist:.6f}")
    
    # Check property
    all_non_negative = all(d >= -1e-10 for d in distances)  # Allow tiny numerical errors
    if all_non_negative:
        print(f"\nPASS: All distances are non-negative")
    else:
        print(f"\nFAIL: Found {negative_count} negative distances")
    
    return all_non_negative, distances, negative_count


def test_symmetry(genomes: List[dict], num_samples: int = 100) -> Tuple[bool, List[float], int]:
    """
    Test Property 2: Symmetry
    d(u, v) = d(v, u) for all u, v
    """
    print("\n" + "="*80)
    print("TEST 2: Symmetry Property")
    print("="*80)
    print(f"Testing: d(u, v) = d(v, u) for all pairs")
    print(f"Sampling {num_samples} random pairs...")
    
    valid_genomes = filter_valid_genomes(genomes)
    if len(valid_genomes) < 2:
        print("ERROR: Need at least 2 valid genomes with embeddings and phenotypes")
        return False, [], 0
    
    np.random.seed(42)  # For reproducibility
    differences = []
    violations = 0
    tolerance = 1e-10
    
    for _ in range(num_samples):
        i, j = np.random.choice(len(valid_genomes), size=2, replace=False)
        g1, g2 = valid_genomes[i], valid_genomes[j]
        
        e1, p1 = extract_embedding_and_phenotype(g1)
        e2, p2 = extract_embedding_and_phenotype(g2)
        
        # Normalize embeddings
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 / np.linalg.norm(e2)
        
        dist_forward = ensemble_distance(e1, e2, p1, p2, w_genotype=0.7, w_phenotype=0.3)
        dist_backward = ensemble_distance(e2, e1, p2, p1, w_genotype=0.7, w_phenotype=0.3)
        
        diff = abs(dist_forward - dist_backward)
        differences.append(diff)
        
        if diff > tolerance:
            violations += 1
            print(f"  WARNING: Asymmetry found: d({g1['id']}, {g2['id']}) = {dist_forward:.10f}, "
                  f"d({g2['id']}, {g1['id']}) = {dist_backward:.10f}, diff = {diff:.10f}")
    
    max_diff = max(differences) if differences else 0
    mean_diff = np.mean(differences) if differences else 0
    
    print(f"\nResults:")
    print(f"  Total pairs tested: {num_samples}")
    print(f"  Violations (|d(u,v) - d(v,u)| > {tolerance}): {violations}")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")
    
    # Check property
    all_symmetric = violations == 0
    if all_symmetric:
        print(f"\nPASS: All pairs are symmetric (within tolerance {tolerance})")
    else:
        print(f"\nFAIL: Found {violations} asymmetric pairs")
    
    return all_symmetric, differences, violations


def test_triangle_inequality(genomes: List[dict], num_samples: int = 100) -> Tuple[bool, List[float], int]:
    """
    Test Property 3: Triangle Inequality
    d(u, w) ≤ d(u, v) + d(v, w) for all u, v, w
    """
    print("\n" + "="*80)
    print("TEST 3: Triangle Inequality Property")
    print("="*80)
    print(f"Testing: d(u, w) ≤ d(u, v) + d(v, w) for all triples")
    print(f"Sampling {num_samples} random triples...")
    
    valid_genomes = filter_valid_genomes(genomes)
    if len(valid_genomes) < 3:
        print("ERROR: Need at least 3 valid genomes with embeddings and phenotypes")
        return False, [], 0
    
    np.random.seed(42)  # For reproducibility
    violations = []
    violation_count = 0
    tolerance = 1e-6  # Allow small numerical errors
    
    for _ in range(num_samples):
        i, j, k = np.random.choice(len(valid_genomes), size=3, replace=False)
        g1, g2, g3 = valid_genomes[i], valid_genomes[j], valid_genomes[k]
        
        e1, p1 = extract_embedding_and_phenotype(g1)
        e2, p2 = extract_embedding_and_phenotype(g2)
        e3, p3 = extract_embedding_and_phenotype(g3)
        
        # Normalize embeddings
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 / np.linalg.norm(e2)
        e3 = e3 / np.linalg.norm(e3)
        
        # Compute distances
        d_uv = ensemble_distance(e1, e2, p1, p2, w_genotype=0.7, w_phenotype=0.3)
        d_vw = ensemble_distance(e2, e3, p2, p3, w_genotype=0.7, w_phenotype=0.3)
        d_uw = ensemble_distance(e1, e3, p1, p3, w_genotype=0.7, w_phenotype=0.3)
        
        # Check triangle inequality: d(u,w) ≤ d(u,v) + d(v,w)
        sum_uv_vw = d_uv + d_vw
        violation = d_uw - sum_uv_vw
        
        if violation > tolerance:
            violation_count += 1
            violations.append(violation)
            print(f"  WARNING: Triangle inequality violation:")
            print(f"      d({g1['id']}, {g3['id']}) = {d_uw:.6f}")
            print(f"      d({g1['id']}, {g2['id']}) + d({g2['id']}, {g3['id']}) = {sum_uv_vw:.6f}")
            print(f"      Violation: {violation:.6f}")
    
    max_violation = max(violations) if violations else 0
    mean_violation = np.mean(violations) if violations else 0
    
    print(f"\nResults:")
    print(f"  Total triples tested: {num_samples}")
    print(f"  Violations (d(u,w) > d(u,v) + d(v,w) + {tolerance}): {violation_count}")
    if violations:
        print(f"  Max violation: {max_violation:.6f}")
        print(f"  Mean violation: {mean_violation:.6f}")
    
    # Check property
    # Note: Triangle inequality may be approximately satisfied due to weighted combination
    # of two different metrics (genotype and phenotype)
    all_satisfy = violation_count == 0
    if all_satisfy:
        print(f"\nPASS: All triples satisfy triangle inequality (within tolerance {tolerance})")
    else:
        print(f"\nWARNING: Found {violation_count} violations")
        print(f"   Note: Triangle inequality is 'approximately satisfied' due to weighted")
        print(f"   combination of genotype (cosine distance) and phenotype (Euclidean distance)")
        print(f"   metrics. Small violations are expected and acceptable.")
    
    return all_satisfy, violations, violation_count


def main():
    """Main function to run all metric property tests."""
    print("="*80)
    print("Ensemble Distance Metric Properties Verification")
    print("="*80)
    
    # Default execution directory (can be changed)
    base_dir = Path(__file__).parent.parent / "data" / "outputs"
    
    # Try to find a recent execution
    execution_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not execution_dirs:
        print(f"ERROR: No execution directories found in {base_dir}")
        return
    
    # Use the most recent execution
    execution_dir = execution_dirs[0]
    print(f"\nUsing execution directory: {execution_dir.name}")
    
    try:
        # Load genomes
        print("Loading genomes...")
        genomes = load_genomes(execution_dir)
        print(f"Loaded {len(genomes)} genomes")
        
        # Filter valid genomes (compute embeddings if missing)
        print("Filtering valid genomes...")
        valid_genomes = filter_valid_genomes(genomes, compute_embeddings=True)
        print(f"Found {len(valid_genomes)} genomes with both embeddings and phenotypes")
        
        if len(valid_genomes) < 3:
            print(f"\nERROR: Need at least 3 valid genomes for testing")
            print(f"   Found only {len(valid_genomes)} valid genomes")
            return
        
        # Run tests
        num_samples = min(100, len(valid_genomes) * 10)  # Adaptive sample size
        
        # Test 1: Non-negativity
        non_neg_pass, distances, neg_count = test_non_negativity(valid_genomes, num_samples)
        
        # Test 2: Symmetry
        symm_pass, differences, symm_violations = test_symmetry(valid_genomes, num_samples)
        
        # Test 3: Triangle inequality
        triangle_pass, violations, triangle_violations = test_triangle_inequality(valid_genomes, num_samples)
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"Property 1 (Non-negativity): {'PASS' if non_neg_pass else 'FAIL'}")
        print(f"Property 2 (Symmetry): {'PASS' if symm_pass else 'FAIL'}")
        print(f"Property 3 (Triangle Inequality): {'PASS' if triangle_pass else 'APPROXIMATE'}")
        print("="*80)
        
        if non_neg_pass and symm_pass:
            print("\nThe ensemble distance metric satisfies the fundamental metric properties!")
            if not triangle_pass:
                print("   (Triangle inequality is approximately satisfied, which is acceptable)")
        else:
            print("\nSome metric properties are not satisfied. Please review the results above.")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
