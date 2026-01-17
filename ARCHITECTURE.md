# System Architecture

Genetic algorithm with semantic speciation (Leader-Follower clustering) for evolving prompts to elicit toxic LLM responses.

## Algorithm

**Fitness**: `f(x) = toxicity_score(LLM(x)) ∈ [0, 1]`

**Population**: `P = E ∪ R`, `|E| = Σᵢ |Sᵢ|`, `|R| ≤ 1000`, `|Sᵢ| ≤ 100`

## Parent Selection

**DEFAULT** (2 parents): `p₁ ~ U(S_random)`, `p₂ ~ U(S_p₁)` → `V = 22`

**EXPLOITATION** (3 parents): `p₁ = argmax_{x∈S_top} f(x)`, `p₂, p₃ ~ U(S_top\{p₁})` → `V = 36`

**EXPLORATION** (3 parents): `p₁ = argmax_{x∈S_top} f(x)`, `p₂ = argmax_{x∈S_j, j≠top} f(x)`, `p₃ = argmax_{x∈S_k, k≠{top,j}} f(x)` → `V = 36`

**Mode Switching**: `mode = DEFAULT if g ≤ 5 else EXPLOIT if Δf/Δg ≤ 0.00 else EXPLORE if stagnation ≥ 5`

## Variation Operators

**10 Mutation**: Informed Evolution, MLM, Paraphrasing, Back Translation, Synonym/Antonym Replacement (POS-aware), Negation, Concept Addition, Typographical Errors, Stylistic Mutation

**2 Crossover**: Semantic Similarity, Semantic Fusion

## Speciation

**Distance Metrics**:
- Genotype: `d_genotype(u,v) = 1 - (e_u · e_v)` where `e_u, e_v ∈ ℝ³⁸⁴` (L2-normalized)
- Phenotype: `d_phenotype(u,v) = ||p_u - p_v||_2 / √8` where `p_u, p_v ∈ [0,1]⁸`
- Ensemble: `d_ensemble = 0.7×d_genotype + 0.3×d_phenotype`

**Clustering**:
- Assignment: `d_ensemble(u, leader) < θ_sim = 0.2`
- Merging: `d_ensemble(leader_i, leader_j) < θ_merge = 0.1`
- Leader: `leader(S_i) = argmax_{x∈S_i} f(x)`

**Operations**:
- Merge: `d_ensemble(leader_i, leader_j) < θ_merge` (only active species merge)
- Freeze: `stagnation(S_i) ≥ 20` → species moved to "frozen" state
- Capacity: Archive excess when `|S_i| > 100` or `|R| > 1000`

**Species States**:
- **active**: Participates in evolution and parent selection
- **frozen**: Stagnated (≥20 generations without improvement), excluded from parent selection but preserved with full data (leader embeddings, distances) for potential merging
- **incubator**: Moved to cluster 0 when `size < min_island_size` (default: < 2), tracked by ID only
  - **Note**: Uses `min_island_size` (min 2), NOT `species_capacity` (max 100)
  - Condition: `actual_size < min_island_size` (strictly less than)

## Process Flow

1. **Generation 0**: Load CSV → Generate responses → Evaluate → Speciate → Distribute
2. **Generation N**: Load population → Select parents → Apply operators → Generate responses → Evaluate → Deduplicate → Speciate → Distribute → Track

**Deduplication**: Two-stage (intra-temp, then population-wide)

## Metrics

**Fitness**: `f_max = max_{x∈P} f(x)`, `f_avg = (1/|P|) Σ_{x∈P} f(x)`

**Diversity**: `D_inter = (1/(K(K-1)/2)) Σ_{i<j} d_ensemble(leader_i, leader_j)`, `D_intra = (1/K) Σᵢ (1/(|Sᵢ|(|Sᵢ|-1)/2)) Σ_{u,v∈Sᵢ} d_ensemble(u,v)`

**Operator Effectiveness** (RQ1):
- `NE = (non_elite_count / calculated_total) × 100` - % variants archived
- `EHR = (elite_count / calculated_total) × 100` - % variants became elites
- `IR = (rejections / calculated_total) × 100` - % variants rejected
- `cEHR = (elite_count / total_variants) × 100` - % valid variants that became elites
- `Δμ = mean(f(v) - f(parent(v)))` - Average fitness change
- `Δσ = std(f(v) - f(parent(v)))` - Std dev of fitness change
- Where `calculated_total = total_variants + rejections + duplicates`

## Files

- `elites.json`: `species_id > 0` (all species members across all generations)
- `reserves.json`: `species_id = 0` (cluster 0, max 1000)
- `archive.json`: Capacity overflow (removed due to limits)
- `temp.json`: Staging (new variants before speciation)
- `EvolutionTracker.json`: Complete evolution history with metrics
- `speciation_state.json`: Species state (active, frozen, incubator) with leader embeddings preserved

## Species Management

**Frozen Species**:
- Preserved with full data: leader embeddings, leader distance, labels, history, and all members
- Members are preserved from when species was active (saved in `member_ids` in speciation_state.json)
- Not included in parent selection (evolution focuses on active species)
- Can merge with active or other frozen species (both are "alive", only difference is parent selection preference)

**Merging**:
- Active and frozen species can merge (frozen species included)
- Requires leader embeddings for distance calculation
- Merged species get new ID, reset stagnation, combine members
- Frozen species that merge are reactivated (moved back to active species)

**Evolution Continuation**:
- If all species freeze, evolution continues using cluster 0 (reserves)
- Cluster 0 is always active and can form new species
- Fallback mechanism selects from all genomes if no active species found
