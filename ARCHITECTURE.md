# ToxSearch with Speciation — Architecture

High-level design of the evolutionary toxicity-search system with leader-follower semantic speciation. For field-by-field definitions see [FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt); for validation and flow details see [experiments/FLOW_AND_VALIDATION.md](experiments/FLOW_AND_VALIDATION.md).

---

## 1. Module Layout

```
src/
├── main.py              # Entry: Gen 0 init, Gen N loop, orchestration
├── ea/                  # Evolutionary algorithm
│   ├── evolution_engine.py   # next_id, create_child, operator dispatch
│   ├── run_evolution.py      # load elites+reserves, parent selection, operators → temp
│   ├── parent_selector.py    # 3-category adaptive tournament (active, reserves, frozen)
│   ├── variation_operators.py
│   └── ... (paraphrasing, crossover, mutation operators)
├── gne/                 # Generate–Evaluate
│   ├── response_generator.py # LLM → generated_output
│   ├── evaluator.py          # Moderation (Google Perspective, hybrid)
│   └── model_interface.py
├── speciation/          # Speciation and distribution
│   ├── run_speciation.py     # process_generation, distribute, EvolutionTracker speciation block
│   ├── leader_follower.py    # Clustering: d_ensemble, θ_sim, θ_merge, cluster0
│   ├── species.py, reserves.py, merging.py, extinction.py
│   ├── embeddings.py, distance.py, phenotype_distance.py
│   ├── metrics.py, genome_tracker.py, labeling.py
│   └── validation.py
└── utils/
    ├── population_io.py      # elites/reserves/archive/temp, EvolutionTracker, calculate_average_fitness
    ├── refusal_penalty.py    # 15% penalty on toxicity for refusals (after evaluation, before speciation)
    ├── refusal_detector.py
    ├── cluster_quality.py    # Silhouette, Davies-Bouldin, Calinski-Harabasz, QD (no Pareto)
    ├── operator_effectiveness.py
    ├── device_utils.py, data_loader.py, config.py
    └── ...
```

---

## 2. High-Level Flow

**Gen 0**

1. Generate (response_generator) → `temp.json`
2. Moderate (evaluator) → `moderation_result`, `north_star_score`
3. **Refusal penalty** (after evaluation, before speciation): 15% reduction on toxicity for refusals; write to `moderation_result` and `north_star_score`
4. **avg_fitness** = `calculate_average_fitness(include_temp=True)` — elites+reserves+temp, before speciation
5. **run_speciation** (process_generation → distribute → update_evolution_tracker_with_speciation)
6. operator_effectiveness(gen=0), calculate_generation_statistics, **update_evolution_tracker_with_statistics**(gen0_stats with avg_fitness)

**Gen N (N ≥ 1)**

1. **run_evolution**: load elites+reserves → **adaptive_tournament_selection** (parents) → operators → `temp.json` (variants)
2. Moderate (temp) → **refusal_penalty**
3. **avg_fitness** = `calculate_average_fitness(include_temp=True)` — before speciation
4. **run_speciation** (temp, gen=N)
5. operator_effectiveness(gen=N), update_evolution_tracker_with_generation_global, update_adaptive_selection_logic
6. calculate_generation_statistics → **update_evolution_tracker_with_statistics**(gen_stats with avg_fitness) → update_population_index_single_file

---

## 3. Parent Selection (3-Category)

- **Category 1:** active species ∪ species 0 (reserves). **Category 2:** frozen species.
- **Rule:** Use Category 2 only when Category 1 has no genomes; if both empty → `RuntimeError`.

**Modes (on chosen category):**

- **DEFAULT:** random species, 2 parents; fill from category if &lt;2.
- **EXPLOIT:** top species by max fitness, 3 parents.
- **EXPLORE:** top + 2 random species, 1 best parent each.

**Max fitness:** actual max over current genomes only (no merge with stored). Same for cluster0 over reserves.

---

## 4. Speciation (process_generation)

1. Load `speciation_state` (if gen&gt;0); snapshot `_prev_max_fitness`.
2. **Phase 1:** Embed temp → leader_follower_clustering (temp + state; `skip_cluster0_outliers=True`) → update `state["species"]`; radius cleanup; capacity; sync cluster0 with reserves.
3. **Phase 2–3:** Merges (θ_merge), check_speciation (cluster0 → new species), extinctions (freeze/incubator).
4. **Phase 4:** Sync `sp.max_fitness` from members; `record_fitness(gen, was_selected, max_fitness_increased)`; freeze/incubator.
5. **Phase 5:** Cluster0 capacity; archive excess.
6. **Phase 6:** `save_to_files_corrected` → elites, reserves, archive.
7. **Phase 7:** Labels, `record_generation`, `save_state` (speciation_state.json), genome_tracker.

**Distribution:** `distribute_genomes(temp)` appends to elites/reserves by `species_id`; dedupe by id. `update_evolution_tracker_with_speciation` writes `gen_entry["speciation"]` (species_count, active/frozen, reserves_size, events, diversity, cluster_quality). **best_fitness and avg_fitness are not in speciation;** they exist only at gen level.

---

## 5. Key Metrics and Conventions

**avg_fitness vs avg_fitness_generation**

- **avg_fitness:** Mean over **old elites + old reserves + all new variants** (elites+reserves+temp) **before speciation**, after evaluation. From `calculate_average_fitness(include_temp=True)` in main. Gen 0: effectively mean(temp). Differs from avg_fitness_generation when genomes are archived this gen.
- **avg_fitness_generation:** Mean over **updated elites + updated reserves** after distribution. From `calculate_generation_statistics` (elites.json + reserves.json only). Archived excluded automatically (removed from those files).

**Refusal penalty**

- Applied after evaluation, before speciation. 15% (×0.85) on toxicity for genomes with `is_refusal(response)==True`. Writes to `moderation_result["google"]["scores"][north_star_metric]` (or legacy `moderation_result["scores"]`); also `genome["north_star_score"]`. `_extract_north_star_score` uses these.

**Cluster quality**

- Silhouette, Davies-Bouldin, Calinski-Harabasz, QD. **Pareto-optimal or multi-objective optimization is not used;** each metric is computed independently.

**EvolutionTracker `gen_entry["speciation"]`**

- Contains: species_count, active_species_count, frozen_species_count, reserves_size; speciation_events, merge_events, extinction_events; archived_count, elites_moved, reserves_moved, genomes_updated; inter/intra_species_diversity, total_population, cluster_quality. **No best_fitness or avg_fitness** (those are at gen level: population_max_toxicity, max_score_variants, avg_fitness).

---

## 6. Output Files (data/outputs/YYYYMMDD_HHMM/)

| File | Role |
|------|------|
| `elites.json` | Species members (species_id &gt; 0), cumulative |
| `reserves.json` | Cluster 0 (species_id = 0), capacity-limited |
| `archive.json` | Archived (capacity overflow, etc.) |
| `temp.json` | Current variants before speciation; cleared/repopulated each gen |
| `EvolutionTracker.json` | Per-gen stats, speciation block, cumulative max, selection state |
| `speciation_state.json` | Species (leader_*, member_ids, max_fitness, stagnation, state), cluster0, metrics |
| `genome_tracker.json` | ID → metadata for lineage |
| `operator_effectiveness_cumulative.csv` | RQ1: per (generation, operator) metrics |
| `figures/` | Fitness, diversity, operator plots |

---

## 7. References

- [FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt) — Field and subfield definitions for all JSON/CSV outputs.
- [experiments/FLOW_AND_VALIDATION.md](experiments/FLOW_AND_VALIDATION.md) — Flow, validation, and field basis (variants vs post-generation vs cumulative).
