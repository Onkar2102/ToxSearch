# System Architecture

Genetic algorithm with semantic speciation (Leader-Follower clustering) for evolving prompts to elicit toxic LLM responses.

## Algorithm

**Fitness**: `f(x) = toxicity_score(LLM(x)) ∈ [0, 1]`

**Population**: `P = E ∪ R`, `|E| = Σᵢ |Sᵢ|`, `|R| ≤ 1000`, `|Sᵢ| ≤ 100`

## Parent Selection

**DEFAULT** (2 parents): `p₁ ~ U(S_random)`, `p₂ ~ U(S_p₁)` → `V = 22`

**EXPLOITATION** (3 parents): `p₁ = argmax_{x∈S_top} f(x)`, `p₂, p₃ ~ U(S_top\{p₁})` → `V = 36`

**EXPLORATION** (3 parents): `p₁ = argmax_{x∈S_top} f(x)`, `p₂ = argmax_{x∈S_j, j≠top} f(x)`, `p₃ = argmax_{x∈S_k, k≠{top,j}} f(x)` → `V = 36`

**Mode Switching**: `mode = DEFAULT if g ≤ 5 else EXPLOIT if Δf/Δg < -0.0001 else EXPLORE if stagnation ≥ 5`

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
- Merge: `d_ensemble(leader_i, leader_j) < θ_merge`
- Freeze: `stagnation(S_i) > 20`
- Capacity: Archive excess when `|S_i| > 100` or `|R| > 1000`

## Process Flow

1. **Generation 0**: Load CSV → Generate responses → Evaluate → Speciate → Distribute
2. **Generation N**: Load population → Select parents → Apply operators → Generate responses → Evaluate → Deduplicate → Speciate → Distribute → Track

**Deduplication**: Two-stage (intra-temp, then population-wide)

## Metrics

**Fitness**: `f_max = max_{x∈P} f(x)`, `f_avg = (1/|P|) Σ_{x∈P} f(x)`

**Diversity**: `D_inter = (1/(K(K-1)/2)) Σ_{i<j} d_ensemble(leader_i, leader_j)`, `D_intra = (1/K) Σᵢ (1/(|Sᵢ|(|Sᵢ|-1)/2)) Σ_{u,v∈Sᵢ} d_ensemble(u,v)`

**Operator Effectiveness**: `NE = 1 - |V_elite|/|V_total|`, `EHR = |V_elite|/|V_total|`, `IR = |V_invalid|/|V_total|`, `Δμ = (1/|V_valid|) Σ_v (f(v) - f(parent(v)))`, `Δσ = √Var({f(v) - f(parent(v))})`

## Files

- `elites.json`: `species_id > 0`
- `reserves.json`: `species_id = 0`
- `archive.json`: Capacity overflow
- `temp.json`: Staging
- `EvolutionTracker.json`: History
- `speciation_state.json`: Species state
