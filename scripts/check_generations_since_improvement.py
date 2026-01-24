#!/usr/bin/env python3
"""
Check that `generations_since_improvement` updates correctly each generation.

For a given outputs directory:
- Computes the per-generation cumulative max toxicity from elites+reserves
- Derives the expected generations_since_improvement sequence
- Compares the final expected value with EvolutionTracker's top-level field
"""

import json
import sys
from pathlib import Path
from typing import Any, List, Dict, Tuple


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_toxicity(genome: Dict[str, Any]) -> float:
    # Prefer explicit toxicity, else north_star_score, else from moderation_result
    if isinstance(genome.get("toxicity"), (int, float)):
        return float(genome["toxicity"])
    if isinstance(genome.get("north_star_score"), (int, float)):
        return float(genome["north_star_score"])
    mr = genome.get("moderation_result") or {}
    if isinstance(mr, dict):
        # Standard shape
        google = mr.get("google", {})
        scores = google.get("scores") or mr.get("scores") or {}
        tox = scores.get("toxicity")
        if isinstance(tox, (int, float)):
            return float(tox)
    return float("nan")


def compute_cumulative_max_by_generation(elites: List[Dict[str, Any]], reserves: List[Dict[str, Any]], max_generation: int) -> List[float]:
    cum_max = []
    current_max = float('-inf')
    # Pre-index genomes by generation
    by_gen: Dict[int, List[Dict[str, Any]]] = {g: [] for g in range(max_generation + 1)}
    for g in elites + reserves:
        gen = g.get("generation")
        if isinstance(gen, int) and 0 <= gen <= max_generation:
            by_gen[gen].append(g)
    # Walk generations, update cumulative max
    for gen in range(max_generation + 1):
        # Consider all genomes with generation <= gen
        toks: List[float] = []
        for g0 in range(gen + 1):
            for genome in by_gen.get(g0, []):
                t = extract_toxicity(genome)
                if isinstance(t, float):
                    toks.append(t)
        if toks:
            current_max = max(current_max, max(toks))
        cum_max.append(current_max if current_max != float('-inf') else float('nan'))
    return cum_max


def derive_generations_since_improvement_sequence(cum_max: List[float]) -> List[int]:
    seq: List[int] = []
    last_improvement_gen = None
    prev = None
    for i, val in enumerate(cum_max):
        if prev is None or (isinstance(val, float) and isinstance(prev, float) and val > prev):
            last_improvement_gen = i
            seq.append(0)
        else:
            if last_improvement_gen is None:
                seq.append(0)
            else:
                seq.append(i - last_improvement_gen)
        prev = val
    return seq


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_generations_since_improvement.py <outputs_path>")
        return 1
    out_path = Path(sys.argv[1])
    tracker = load_json(out_path / "EvolutionTracker.json")
    elites = load_json(out_path / "elites.json") or []
    reserves = load_json(out_path / "reserves.json") or []
    if not isinstance(tracker, dict) or not isinstance(elites, list) or not isinstance(reserves, list):
        print("Error: Invalid files in outputs path")
        return 1

    generations = tracker.get("generations", [])
    max_generation = 0
    for gen in generations:
        gn = gen.get("generation_number", 0)
        if isinstance(gn, int) and gn > max_generation:
            max_generation = gn

    cum_max = compute_cumulative_max_by_generation(elites, reserves, max_generation)
    seq = derive_generations_since_improvement_sequence(cum_max)

    top_level_gsi = tracker.get("generations_since_improvement")
    expected_final = seq[-1] if seq else None

    print("=== Generations Since Improvement Check ===")
    print(f"Run: {out_path.name}")
    print(f"Max generation: {max_generation}")
    print("Cumulative max toxicity by generation:")
    for i, v in enumerate(cum_max):
        print(f"  Gen {i}: {v}")
    print("Expected generations_since_improvement sequence:")
    print("  " + ", ".join(str(x) for x in seq))
    print(f"Tracker top-level generations_since_improvement: {top_level_gsi}")
    print(f"Expected final value: {expected_final}")
    if expected_final == top_level_gsi:
        print("✓ MATCH: Tracker value matches expected final")
        return 0
    else:
        print("✗ MISMATCH: Tracker value does not match expected final")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
