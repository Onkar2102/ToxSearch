================================================================================
PROCESS FLOW: ToxSearch with Speciation
================================================================================

This document describes the complete end-to-end process flow of the evolutionary
text generation system with dynamic speciation.

--------------------------------------------------------------------------------
INITIALIZATION (Before Generation 0)
--------------------------------------------------------------------------------

1. System Setup
   - Initialize logging and output directory (data/outputs/YYYYMMDD_HHMM/)
   - Load device configuration and model paths
   - Update RGConfig.yaml and PGConfig.yaml with model paths
   - Initialize response generator (RG) and prompt generator (PG) models
   - Load seed prompts from data/prompt.csv

2. Generation 0 Initial Population
   - Generate initial responses using response_generator.process_population()
   - Output: temp.json (genomes with prompts and generated_output)
   - Evaluate responses using moderation APIs (Google Perspective API)
   - Apply refusal penalties (15% reduction on toxicity for refusals)
   - Calculate avg_fitness = mean(temp.json fitness) [before speciation, after evaluation]

3. Speciation Initialization (Generation 0)
   - Initialize SpeciationConfig with parameters (theta_sim, theta_merge, etc.)
   - Compute embeddings for all genomes in temp.json
   - Run leader-follower clustering to form initial species
   - Distribute genomes to elites.json (species_id > 0) and reserves.json (species_id = 0)
   - Archive excess genomes to archive.json (capacity limits)
   - Save speciation_state.json with initial species structure

4. Generation 0 Statistics
   - Calculate operator effectiveness metrics (empty for gen 0, no operators)
   - Calculate generation statistics (elites_count, reserves_count, etc.)
   - Update EvolutionTracker.json with gen 0 entry
   - Initialize cumulative budget tracking

--------------------------------------------------------------------------------
EVOLUTION LOOP (Generation N, N ≥ 1)
--------------------------------------------------------------------------------

For each generation N:

PHASE 1: Variant Generation
---------------------------
1. Load Population
   - Load elites.json and reserves.json
   - Load speciation_state.json (if exists)

2. Parent Selection (Adaptive)
   - Category 1: Active species ∪ reserves (species_id = 0)
   - Category 2: Frozen species (only if Category 1 is empty)
   - Selection mode determined by adaptive logic:
     * DEFAULT: Random species, 2 parents
     * EXPLOIT: Top species by fitness, 3 parents (when slope_of_avg_fitness <= 0)
     * EXPLORE: Top + 2 random species, 1 parent each (when stagnation detected)

3. Variant Creation
   - Apply variation operators (mutation/crossover) to selected parents
   - Operators: LLMBasedParaphrasing, ConceptAddition, BackTranslation, etc.
   - Track operator statistics (rejections, duplicates)
   - Output: temp.json (new variants with operator, variant_type, parents)

4. Response Generation
   - Generate responses for all variants in temp.json
   - Update temp.json with generated_output, model_name, response_duration

PHASE 2: Evaluation
-------------------
1. Moderation
   - Evaluate all variants using moderation APIs
   - Update temp.json with moderation_result, toxicity, evaluation_duration

2. Refusal Penalty
   - Detect refusals in generated_output
   - Apply 15% penalty (×0.85) to toxicity for refusals
   - Update moderation_result and north_star_score

3. Pre-Speciation Metrics
   - Calculate avg_fitness = mean(old elites + old reserves + all new variants)
     [BEFORE speciation, AFTER evaluation]
   - Calculate variant statistics from temp.json:
     * max_score_variants: max fitness in temp.json
     * min_score_variants: min fitness in temp.json
     * avg_fitness_variants: mean fitness in temp.json
     * variants_created: total generated (remaining + duplicates + rejections)
     * mutation_variants: count with variant_type=="mutation"
     * crossover_variants: count with variant_type=="crossover"

PHASE 3: Speciation (process_generation)
----------------------------------------
The speciation process has 8 phases:

Phase 1: Existing Species Processing
  1. Compute embeddings for temp.json genomes
  2. Process variants against existing species (leader-follower clustering)
     - Skip cluster 0 outliers (Flow 1)
  3. Generation 0 ONLY: Immediate capacity enforcement after species formation
  4. Sync cluster 0 with reserves.json
  5. Generation N: Radius cleanup of existing species (after all variants processed)
  6. Save intermediate state #1

Phase 2: Cluster 0 Speciation (Isolated)
  1. Load cluster 0 from reserves.json
  2. Apply isolated cluster 0 speciation (Flow 2)
     - Form new species from reserves when min_cluster_size reached
  3. Save intermediate state #2

Phase 3: Merging
  1. Merge similar species (theta_merge threshold)
  2. No radius/capacity enforcement during merging
  3. Save intermediate state (after merging)

Phase 4: Radius & Capacity Enforcement
  1. Radius enforcement for ALL species (existing + newly formed + merged)
  2. Capacity enforcement for ALL species (excess → archive.json)
  3. Validate no duplicate leader IDs
  4. Save intermediate state (after enforcement)

Phase 5: Freeze & Incubator
  1. Sync max_fitness to actual max over current members
  2. Record fitness history and stagnation
     - Stagnation: reset if max_fitness increased, else +=1 if was_selected
  3. Freeze stagnant species (species_stagnation threshold)
  4. Move small species to incubator (min_island_size threshold)
  5. Save intermediate state #3

Phase 6: Cluster 0 Capacity Enforcement
  1. Enforce cluster 0 capacity at end
  2. Archive excess reserves to archive.json
  3. Save intermediate state (after cluster 0 capacity)

Phase 7: Final Corrected Save
  1. Save corrected files (elites.json, reserves.json, archive.json)
  2. Use genome_tracker as authoritative source of truth

Phase 8: Update Metrics & Stats
  1. Update metrics from corrected files
  2. Calculate cluster quality metrics (silhouette, Davies-Bouldin, etc.)

PHASE 4: Distribution (phase8_redistribute_genomes)
----------------------------------------------------
After process_generation completes:
1. Redistribute genomes using genome_tracker as source of truth
2. Update elites.json, reserves.json, archive.json
3. Update speciation_state.json with final state

PHASE 5: Post-Speciation Processing
-----------------------------------
1. Operator Effectiveness Metrics
   - Calculate NE, EHR, IR, cEHR, Δμ, Δσ for each operator
   - Save to operator_effectiveness_cumulative.csv
   - Generate visualizations (if data available)

2. Update EvolutionTracker with Generation Data
   - Update variant counts (variants_created, mutation_variants, crossover_variants)
   - Preserve speciation data from process_generation

3. Adaptive Selection Logic Update
   - Calculate slope_of_avg_fitness from avg_fitness_history
   - Update selection_mode based on:
     * EXPLOIT: slope <= 0
     * EXPLORE: generations_since_improvement >= stagnation_limit
     * DEFAULT: otherwise
   - Update generations_since_improvement

4. Generation Statistics
   - Calculate comprehensive statistics:
     * elites_count, reserves_count, archived_count
     * avg_fitness_elites, avg_fitness_reserves, avg_fitness_generation
     * population_max_toxicity (cumulative max)
   - Update EvolutionTracker with all statistics

5. Visualizations
   - Generate live analysis visualizations
   - Update figures/ directory

6. Population Index Update
   - Update EvolutionTracker.total_generations
   - Update cumulative budget

--------------------------------------------------------------------------------
TERMINATION
--------------------------------------------------------------------------------

Evolution stops when:
1. max_generations reached (if specified)
2. Threshold achieved (population_max_toxicity >= north_star_threshold)
3. Runtime error or user interruption

Final steps:
- Save final speciation_state.json
- Save final EvolutionTracker.json
- Generate final visualizations
- Log total execution time and statistics

--------------------------------------------------------------------------------
KEY METRICS AND THEIR TIMING
--------------------------------------------------------------------------------

avg_fitness:
  - Calculated BEFORE speciation, AFTER evaluation
  - Formula: mean(old elites + old reserves + all new variants in temp.json)
  - Used for: adaptive selection logic, slope calculation

avg_fitness_generation:
  - Calculated AFTER distribution
  - Formula: mean(updated elites + updated reserves)
  - Archived genomes excluded automatically

max_score_variants:
  - Calculated from temp.json BEFORE speciation
  - Represents max fitness among variants created this generation
  - Used to track generation-level improvements

population_max_toxicity:
  - Cumulative max of best_fitness across all generations
  - Updated after each generation
  - Used for threshold checking and stagnation detection

slope_of_avg_fitness:
  - Calculated from avg_fitness_history (sliding window)
  - Used to determine selection mode (EXPLOIT when slope <= 0)

--------------------------------------------------------------------------------
FILE UPDATES PER GENERATION
--------------------------------------------------------------------------------

temp.json:
  - Created/updated: Variant generation, response generation, evaluation
  - Cleared/updated: During speciation (embeddings added, then removed)
  - Final state: May contain leftovers after distribution

elites.json:
  - Updated: After distribution (Phase 8)
  - Contains: All genomes with species_id > 0
  - Cumulative: All elites from all generations

reserves.json:
  - Updated: After distribution (Phase 8)
  - Contains: All genomes with species_id = 0
  - Cumulative: All reserves from all generations
  - Sorted: By toxicity (descending)

archive.json:
  - Updated: During capacity enforcement, cluster 0 capacity enforcement
  - Contains: Archived genomes (capacity overflow, non-elite, etc.)
  - Cumulative: All archived genomes

speciation_state.json:
  - Updated: After each speciation phase (intermediate saves)
  - Final update: After Phase 8 distribution
  - Contains: Species structure, cluster 0 state, metrics, config

EvolutionTracker.json:
  - Updated: Multiple times per generation
    * After speciation (speciation data)
    * After variant generation (variant counts)
    * After statistics calculation (comprehensive stats)
  - Contains: Per-generation entries, cumulative metrics, budget

genome_tracker.json:
  - Updated: During speciation (events tracked)
  - Contains: Event log of genome movements (clustering, archiving, merging)

operator_effectiveness_cumulative.csv:
  - Updated: After each generation
  - Contains: Per-operator metrics (NE, EHR, IR, cEHR, Δμ, Δσ)

--------------------------------------------------------------------------------
END OF PROCESS FLOW
--------------------------------------------------------------------------------
