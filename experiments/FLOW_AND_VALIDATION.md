# ToxSearch with Speciation: End-to-End Flow and Validation

## 1. High-Level Flow (per generation)

```
Gen 0:
  generate (RG) → temp.json → moderate → refusal_penalty
  → avg_fitness = calculate_average_fitness(include_temp=True)  # before speciation, after evaluation
  → run_speciation(temp, gen=0) [process_generation → distribute → update_evolution_tracker_with_speciation]
  → operator_effectiveness(gen=0) → calculate_generation_statistics → update_evolution_tracker_with_statistics(gen0_stats)

Gen N (N≥1):
  run_evolution: load elites+reserves → adaptive_tournament_selection (parents) → operators → temp (variants)
  → moderate (temp) → refusal_penalty
  → avg_fitness = calculate_average_fitness(include_temp=True)  # before speciation, after evaluation
  → run_speciation(temp, gen=N) [process_generation → distribute → update_evolution_tracker_with_speciation]
  → operator_effectiveness(gen=N)
  → update_evolution_tracker_with_generation_global (variant counts, preserves speciation)
  → update_adaptive_selection_logic
  → calculate_generation_statistics → update_evolution_tracker_with_statistics(gen_stats)
  → update_population_index_single_file
```

## 2. Parent Selection (3-Category)

- **Category 1:** `(all_species_in_genomes - frozen) | {0}` — active species + species 0 (reserves).  
- **Category 2:** frozen species.  
- **Rule:** Use Category 2 only when Category 1 has no genomes. If both empty → `RuntimeError` to end the run.

**Modes (on the chosen category):**

- **DEFAULT:** random species from sorted, 2 parents; if chosen has &lt;2, fill from category.  
- **EXPLOIT:** top species by max fitness, 3 parents; if &lt;3, fill from category.  
- **EXPLORE:** top + 2 random species, 1 best parent each; if &lt;3 species, reuse/fill from category.

**Max fitness:** actual max over current genomes only (no merge with stored). Same for cluster0 over reserves.

## 3. Speciation (process_generation)

1. **Load** speciation_state (if gen&gt;0), snapshot `_prev_max_fitness` (before Phase 1).
2. **Phase 1:** Embed temp → `leader_follower_clustering` (temp + speciation_state; `skip_cluster0_outliers=True`) → `state["species"]` updated; radius cleanup; capacity; sync cluster0 with reserves.
3. **Phase 2–3:** Merges, check_speciation (cluster0 → new species), extinctions (freeze / incubator).
4. **Phase 4:** Sync `sp.max_fitness` from members; `record_fitness(gen, was_selected, max_fitness_increased)`; freeze / incubator.
5. **Phase 5:** Cluster0 capacity; archive excess.
6. **Phase 6:** `save_to_files_corrected` → elites, reserves, archive (from state + existing files, correct membership).
7. **Phase 7:** Labels, `record_generation` (metrics from elites + reserves paths), `save_state` (speciation_state.json), genome_tracker.
8. **Return** (species, cluster0). Distribution is done in `run_speciation`, not inside `process_generation`.

## 4. Distribution and run_speciation Main Path

- `process_generation(genomes from temp)` runs the pipeline above.  
- `_update_genomes_with_species(genomes)` (from temp, species_id from state).  
- `distribute_genomes(temp)`: append temp’s genomes to elites/reserves by `species_id`; dedupe by id.  
- `_update_speciation_state_cluster0_size_after_distribution`.  
- `update_evolution_tracker_with_speciation(result, speciation_stats)` writes the full **speciation** block.  
- `remove_embeddings_from_temp`.

## 5. EvolutionTracker Updates

- **From `update_evolution_tracker_with_speciation` (run_speciation):**  
  Full `gen_entry["speciation"]` (species_count, active/frozen, reserves_size, events, diversity, cluster_quality).  
  best_fitness and avg_fitness are not in speciation; they are only at gen level.  
  Runs only when the tracker file exists; for Gen 0 the tracker is created later by main, so it can skip.

- **From `update_evolution_tracker_with_statistics` (main, population_io):**  
  Top-level: elites_count, reserves_count, archived_count, total_population, avg_fitness_*, variant stats, operator_statistics.  
  - If `gen_entry["speciation"]` exists and has `species_count` → keep it.  
  - Else, if `statistics` has `species_count` or `reserves_size` → build a minimal speciation block from `statistics` (so Gen 0 and edge cases are covered).

- **From `update_evolution_tracker_with_generation_global` (main, run_evolution):**  
  genome_id, max_score_variants, variant counts, avg_fitness; it **preserves** existing `speciation`.

## 6. speciation_state.json

- **Per species:** id, leader_*, member_ids, size (= len(member_ids)), max_fitness (actual over members), stagnation, species_state, etc.  
- **cluster0:** size, cluster0_size_from_reserves, max_fitness (actual over reserves, no merge).  
- **metrics:** from `metrics_tracker`.  
- **save_state** runs after Phase 6/7; member_ids and size are aligned with elites (and, where used, with state).

## 7. operator_effectiveness_cumulative.csv

- `calculate_table4_metrics` uses elites, reserves, archive, temp, and operator_statistics.  
- `save_operator_effectiveness_cumulative` appends/overwrites by generation.  
- Fields: generation, operator, NE, EHR, IR, cEHR, Δμ, Δσ, total_variants, elite_count, non_elite_count, rejections, duplicates.

## 8. Checks on data/outputs/20260118_2218

- **Validation (comprehensive_validation.py):** 0 issues; warnings: logs dir, and `cluster_quality.num_clusters` vs `active_species` (cluster_quality is from embedding-based clustering; active_species from speciation_state; by design they can differ).  
- **Gen 1 speciation:** `species_count=0` in that run can be from an older version or from all new variants going to reserves (e.g. with `skip_cluster0_outliers=True` and nothing close to the single gen‑0 species). Worth re-running to confirm with current code.  
- **EvolutionTracker / speciation:** With the new fallback in `update_evolution_tracker_with_statistics`, Gen 0 and cases where `update_evolution_tracker_with_speciation` does not run get a minimal speciation block from `statistics` (species_count, reserves_size, events, etc.).  
- **Main:** Now passes `active_species_count`, `frozen_species_count`, `elites_moved`, `reserves_moved`, `genomes_updated` from `speciation_result` into `gen_stats` so the fallback block can be fully filled when used.

## 9. Field consistency

- **elites_count, reserves_count, total_population:** From `calculate_generation_statistics` (reads elites.json, reserves.json after distribution).  
- **speciation.species_count, reserves_size:** From run_speciation’s `update_evolution_tracker_with_speciation` when it runs; else from `update_evolution_tracker_with_statistics` fallback using `speciation_result`/`gen_stats`.  
- **cluster0.max_fitness:** Set in `save_state` from `max` over current reserves only (no merge).  
- **species.max_fitness:** Actual max over current members; on load in leader_follower from members only; in Phase 4 synced before `record_fitness`.

**avg_fitness_generation vs avg_fitness:**

- **avg_fitness:** Mean over **old elites + old reserves + all new variants** (elites + reserves + temp) **before speciation**, after evaluation (and refusal penalty). From `calculate_average_fitness(include_temp=True)` in main. Gen 0: elites/reserves empty, so effectively mean(temp).
- **avg_fitness_generation:** Mean over **updated elites + updated reserves** (elites.json + reserves.json) **after distribution**. From `calculate_generation_statistics`. Genomes moved to archive are automatically excluded — stats are computed from elites and reserves files only; archived have been removed from those files. Same as avg_fitness when no archiving this gen; differs when some genomes are archived this generation.

*Implementation:* `calculate_generation_statistics` computes `avg_fitness_generation` only; `avg_fitness` is supplied by main from `calculate_average_fitness(include_temp=True)`. `update_evolution_tracker_with_statistics` uses `statistics["avg_fitness"]` when provided, else `avg_fitness_generation`. The slope (`slope_of_avg_fitness`) is over `gen["avg_fitness"]`; `update_adaptive_selection_logic` receives `current_gen_avg_fitness` from main (same as `avg_fitness`) so the slope uses avg_fitness consistently for all generations.

---

## 10. What each field is based on: variants-only vs post-generation vs cumulative

- **Variants (this generation):** Only variants created and processed in that generation (temp, or `generation == current_generation` in elites/reserves/archive, or operator/run stats for this gen).
- **Post-generation (all genomes):** Snapshot of the full population (elites + reserves, and archive where relevant) **after** this generation's variants are merged and distributed.
- **Cumulative:** Running totals or history across generations (tracker-level max, history arrays, or a file aggregating all generations).

### 10.1 EvolutionTracker.json

**Per-generation entry (`generations[i]`):**

| Field | Basis | Notes |
|-------|--------|------|
| `variants_created` | **Variants (this gen)** | Total generated by operators before dedup/rejection. |
| `mutation_variants`, `crossover_variants` | **Variants (this gen)** | Counts in temp after dedup/rejection for this gen. |
| `max_score_variants`, `min_score_variants`, `avg_fitness_variants` | **Variants (this gen)** | From temp.json (this gen's variants) before speciation; main overwrites with pre-speciation values. |
| `genome_id` | **Variants (this gen)** | Best genome in population with `generation == gen_number`, or fallback from parents. |
| `operator_statistics` | **Variants (this gen)** | Rejections/duplicates per operator for this gen. |
| `budget` (llm_calls, api_calls, *time) | **Variants (this gen)** | From genomes with `generation == current_generation`. |
| `elites_count`, `reserves_count`, `total_population`, `archived_count` | **Post-generation (all genomes)** | `calculate_generation_statistics`: elites/reserves/archive with `generation <= current_generation` = full population at end of gen. |
| `avg_fitness_generation` | **Post-generation (all genomes)** | **After distribution:** mean over updated elites + reserves (from files). Archived excluded (not in elites/reserves). |
| `avg_fitness` | **Before speciation, after evaluation** | Mean over old elites + old reserves + all new variants (elites+reserves+temp); from `calculate_average_fitness(include_temp=True)` in main before `run_speciation`. Differs from avg_fitness_generation when archiving this gen. |
| `avg_fitness_elites`, `avg_fitness_reserves` | **Post-generation (all genomes)** | Over elites and reserves (up to `current_generation`) after distribution. |

**`gen_entry["speciation"]` block:**

| Field | Basis | Notes |
|-------|--------|------|
| `speciation_events`, `merge_events`, `extinction_events` | **Variants (this gen)** | Events during this gen's speciation run. |
| `archived_count` (in speciation) | **Variants (this gen)** | `state["_archived_count"]` = archived during this gen. |
| `elites_moved`, `reserves_moved`, `genomes_updated` | **Variants (this gen)** | From `distribute_genomes` / `_update_genomes_with_species` for this gen's temp. |
| `species_count`, `active_species_count`, `frozen_species_count`, `reserves_size` | **Post-generation (all genomes)** | State of species and reserves after this gen. |
| `inter_species_diversity`, `intra_species_diversity`, `total_population`, `cluster_quality` | **Post-generation (all genomes)** | From `record_generation` over elites + reserves (and species) after Phase 6. best_fitness and avg_fitness are not in speciation; they are at gen level only. |

**Tracker-level (top-level):**

| Field | Basis | Notes |
|-------|--------|------|
| `population_max_toxicity` | **Cumulative** | Max over all generations. |
| `avg_fitness_history` | **Cumulative** | Sliding window over last N generations' `avg_fitness`. |
| `slope_of_avg_fitness` | **Cumulative** | From `avg_fitness_history`. |
| `selection_mode`, `generations_since_improvement` | **Cumulative** | Adaptive state over the run. |
| `speciation_summary` | **Cumulative** | Updated each gen; reflects latest state. |
| `cumulative_budget` | **Cumulative** | Sum of each gen's `budget`. |

### 10.2 speciation_state.json

**Per-species and cluster0:**

| Field | Basis | Notes |
|-------|--------|------|
| `member_ids`, `size`, `max_fitness`, `min_fitness`, `stagnation`, `species_state`, `leader_*`, etc. | **Post-generation (all genomes)** | Species/cluster0 at save time after this gen. |
| `cluster0.size`, `cluster0_size_from_reserves`, `cluster0.max_fitness` | **Post-generation (all genomes)** | From reserves after distribution. |

**`metrics` (from `metrics_tracker.to_dict()`):**

| Field | Basis | Notes |
|-------|--------|------|
| `history` | **Cumulative** | One entry per generation. |
| Each history entry (best, avg, diversity, etc.) | **Post-generation (all genomes)** | `record_generation` over elites + reserves at end of that gen. |

**`incubators`:** **Post-generation (all genomes)** — IDs in incubator at save time.

### 10.3 operator_effectiveness_cumulative.csv

- **File:** **Cumulative** — one row per `(generation, operator)` across all generations.
- **Each row's metrics:** **Variants (this generation) only** for that operator:

| Field | Basis | Notes |
|-------|--------|------|
| `generation`, `operator` | — | Row identifier. |
| `total_variants`, `elite_count`, `non_elite_count` | **Variants (this gen)** | From elites/reserves/archive with `generation == current_generation`; elite/non-elite by `initial_state`. |
| `rejections`, `duplicates` | **Variants (this gen)** | From `operator_statistics` for this gen. |
| `NE`, `EHR`, `IR`, `cEHR` | **Variants (this gen)** | Derived from the above for this gen. |
| `Δμ`, `Δσ` | **Variants (this gen)** | Over this gen's variants only (current_toxicity − parent_score). |
