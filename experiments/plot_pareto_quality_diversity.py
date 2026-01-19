#!/usr/bin/env python3
"""
Post-hoc Pareto plot: Quality vs Diversity for ToxSearch with Speciation.

Extracts from EvolutionTracker.json (no live tracking needed):
- quality:  population_max_toxicity (cumulative max over all generations of best
            toxicity in elites+reserves per generation)
- diversity: inter_species_diversity from the last generation's speciation block
            (mean pairwise distance between all species leaders, active+frozen)

Usage:
  python experiments/plot_pareto_quality_diversity.py --base data/outputs --runs 20260119_1407 20260119_1342
  python experiments/plot_pareto_quality_diversity.py --base data/outputs --runs 20260118_2218 --out-dir . --per-gen

Outputs:
  - pareto_quality_diversity.png
  - pareto_quality_diversity.csv (run_id, quality, diversity, species_count, ...)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _find_base() -> Path:
    p = Path(__file__).resolve().parent
    # experiments/ -> project root
    root = p.parent
    return root / "data" / "outputs"


def load_run_point(base: Path, run_id: str) -> dict | None:
    """Load (quality, diversity) and related fields for one run from EvolutionTracker.json."""
    et_path = base / run_id / "EvolutionTracker.json"
    if not et_path.exists():
        return None
    try:
        with open(et_path, "r", encoding="utf-8") as f:
            et = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load {et_path}: {e}", file=sys.stderr)
        return None

    quality = float(et.get("population_max_toxicity") or 0.0)

    gens = et.get("generations") or []
    diversity = 0.0
    intra = 0.0
    species_count = 0
    total_population = 0
    last_gen = None

    if gens:
        last = gens[-1]
        last_gen = int(last.get("generation_number", -1))
        spec = last.get("speciation")
        if spec is not None and isinstance(spec, dict):
            diversity = float(spec.get("inter_species_diversity") or 0.0)
            intra = float(spec.get("intra_species_diversity") or 0.0)
            species_count = int(spec.get("species_count") or 0)
            total_population = int(spec.get("total_population") or 0)

    return {
        "run_id": run_id,
        "quality": round(quality, 4),
        "diversity": round(diversity, 4),
        "intra_species_diversity": round(intra, 4),
        "species_count": species_count,
        "total_population": total_population,
        "last_generation": last_gen,
        "total_generations": int(et.get("total_generations") or 0),
    }


def load_per_generation_points(base: Path, run_id: str) -> list[dict]:
    """Load one point per (run, generation). Quality uses running max of max_score_variants."""
    et_path = base / run_id / "EvolutionTracker.json"
    if not et_path.exists():
        return []
    try:
        with open(et_path, "r", encoding="utf-8") as f:
            et = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load {et_path}: {e}", file=sys.stderr)
        return []

    gens = et.get("generations") or []
    running_max = 0.0
    out = []
    for g in gens:
        gen_num = g.get("generation_number", 0)
        mv = float(g.get("max_score_variants") or 0.0)
        running_max = max(running_max, mv)
        spec = g.get("speciation")
        div = 0.0
        if spec is not None and isinstance(spec, dict):
            div = float(spec.get("inter_species_diversity") or 0.0)
        out.append({
            "run_id": run_id,
            "generation": gen_num,
            "quality": round(running_max, 4),
            "diversity": round(div, 4),
        })
    return out


def pareto_frontier(points: list[dict], quality_key: str = "quality", diversity_key: str = "diversity") -> list[dict]:
    """Non-dominated points when maximizing both quality and diversity."""
    if not points:
        return []
    Q = np.array([p[quality_key] for p in points])
    D = np.array([p[diversity_key] for p in points])
    dominated = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        if dominated[i]:
            continue
        # i is dominated if some j has Q[j]>=Q[i] and D[j]>=D[i] with at least one strict
        for j in range(len(points)):
            if i == j or dominated[j]:
                continue
            if (Q[j] >= Q[i] and D[j] >= D[i]) and (Q[j] > Q[i] or D[j] > D[i]):
                dominated[i] = True
                break
    return [p for i, p in enumerate(points) if not dominated[i]]


def plot_pareto(
    points: list[dict],
    frontier: list[dict],
    out_path: Path,
    *,
    title: str = "Pareto: Quality vs Diversity (Speciation)",
    quality_label: str = "Quality (population_max_toxicity)",
    diversity_label: str = "Diversity (inter_species_diversity)",
) -> None:
    if not HAS_MPL:
        print("matplotlib not available; skipping plot.", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    q = [p["quality"] for p in points]
    d = [p["diversity"] for p in points]
    run_ids = [p.get("run_id", "") for p in points]
    ax.scatter(d, q, c="#2ecc71", s=60, alpha=0.8, edgecolors="darkgreen", label="Runs", zorder=2)
    for i, rid in enumerate(run_ids):
        ax.annotate(rid, (d[i], q[i]), fontsize=7, alpha=0.9, xytext=(4, 4), textcoords="offset points")
    if frontier:
        fd = [p["diversity"] for p in frontier]
        fq = [p["quality"] for p in frontier]
        idx = np.lexsort(([-x for x in fq], fd))  # sort by diversity asc, then quality desc
        fd = [fd[i] for i in idx]
        fq = [fq[i] for i in idx]
        ax.plot(fd, fq, "k--", alpha=0.7, linewidth=1.5, label="Pareto frontier", zorder=1)
    ax.set_xlabel(diversity_label)
    ax.set_ylabel(quality_label)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Post-hoc Pareto: Quality vs Diversity from EvolutionTracker.json")
    ap.add_argument("--base", type=Path, default=None, help="Base outputs dir (default: data/outputs)")
    ap.add_argument("--runs", nargs="*", default=[], help="Run directory names (e.g. 20260119_1407)")
    ap.add_argument("--out-dir", type=Path, default=Path("comparison_results"), help="Output directory")
    ap.add_argument("--per-gen", action="store_true", help="Plot one point per (run, generation) and running-max quality")
    ap.add_argument("--no-plot", action="store_true", help="Only write CSV, no plot")
    args = ap.parse_args()

    base = args.base or _find_base()
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    if not base.exists():
        print(f"Base path does not exist: {base}", file=sys.stderr)
        sys.exit(1)

    runs = args.runs
    if not runs:
        print("Provide at least one --runs run ID.", file=sys.stderr)
        sys.exit(1)

    if args.per_gen:
        all_pts = []
        for rid in runs:
            all_pts.extend(load_per_generation_points(base, rid))
        points = all_pts
        frontier = pareto_frontier(points) if points else []
    else:
        points = []
        for rid in runs:
            p = load_run_point(base, rid)
            if p is not None:
                points.append(p)
        frontier = pareto_frontier(points) if points else []

    if not points:
        print("No data extracted. Check --base and --runs.", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "pareto_quality_diversity.csv"

    # CSV
    if points:
        import csv
        keys = list(points[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(points)
        print(f"Saved {csv_path} ({len(points)} rows)")

    if HAS_MPL and not args.no_plot:
        plot_pareto(
            points,
            frontier,
            args.out_dir / "pareto_quality_diversity.png",
            title="Pareto: Quality vs Diversity (Speciation)" + (" (per generation)" if args.per_gen else " (per run)"),
        )


if __name__ == "__main__":
    main()
