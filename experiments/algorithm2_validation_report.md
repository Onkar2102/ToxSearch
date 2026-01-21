# Algorithm 2: Initial Species and Reserve Formation — Validation vs Implementation

## Summary

**Algorithm 2** (from the screenshot) describes **initial** species and reserve formation for generation 0: form species from a pool **P** by repeatedly taking the most toxic prompt as seed, gathering similar prompts, and either creating a species (if ≥ C_min) or sending only the seed to reserves. It also enforces species capacity by archiving the least toxic in each species.  

The implementation uses **`cluster0_speciation_isolated`** (and, for Gen 0 only, equivalent logic in **`leader_follower_clustering`**) on **`state["cluster0"].individuals`** (reserves + temp unassigned). The **overall structure and intent match**, but there are **specific differences** in: (1) **leader choice**, (2) **member set**, (3) **behavior when |S_i| < C_min**, (4) **where C_species is applied**, and (5) **extra pre-condition and lack of C_reserves in the algorithm**.

---

## 1. Parameter and input mapping

| Algorithm 2 | Implementation | Notes |
|-------------|----------------|-------|
| **P** | `state["cluster0"].individuals` with embeddings | For Gen 0 / Phase 2: reserves + temp unassigned (species_id=0, with prompt_embedding), after sync and add-from-temp. ✓ |
| **θ_sim** | `config.theta_sim` | ✓ |
| **C_min** | `config.min_island_size` | ✓ |
| **C_species** | `config.species_capacity` | ✓ (Algorithm 2’s “Cmax”) |
| **C_reserves** | `config.cluster0_max_capacity` | In Algorithm 2 this is required but **not used** in the steps. In the impl, C_reserves is enforced in **Phase 5**, not in `cluster0_speciation_isolated`. |

---

## 2. Step-by-step comparison

### 2.1 Initialization (S←∅, R←∅, A←∅)

- **Algorithm 2:** Explicit sets S, R, A.
- **Impl:** Species are created and added to `state["species"]`; R is `state["cluster0"]` (pre-filled); archive is handled by `_archive_individuals` and genome tracker. Conceptually aligned.

---

### 2.2 Main loop: “While P non-empty” vs single pass

- **Algorithm 2:**  
  - Sort P by **t_toxicity descending**.  
  - **P_0** = first (most toxic). **S_i = {P_0}**.  
  - For each other **P_j**: if **d_ensemble(P_j, P_0) < θ_sim** then add P_j to S_i.  
  - If **|S_i| < C_min**: **R ← R ∪ {P_0}**, remove P_0 from P (others stay in P).  
  - Else: **S ← S ∪ {S_i}**, **P ← P \ S_i**.  
  - Repeat until P is empty.

- **Impl (`cluster0_speciation_isolated`):**  
  - **Single pass** in fitness-descending order (fitness ∝ toxicity).  
  - First individual = first potential leader; each remaining either joins the **nearest** potential leader (if **d < θ_sim**) or becomes a new potential leader.  
  - When a potential leader has **1 + |followers| ≥ min_island_size**, the impl:
    - Picks **leader = argmax fitness** over {potential_leader} ∪ followers (Algorithm 2 keeps **P_0**).
    - Builds **valid_members** = {m : d_ensemble(m, leader) < θ_sim} (Algorithm 2 uses **d(P_j, P_0) < θ_sim** only).
  - If **|valid_members| ≥ min_island_size**: form species, **remove_batch(valid_members)** from cluster0.  
  - If **|valid_members| < min_island_size**: **do not** form a species; the whole candidate set **stays in cluster0** (no one is moved to R; the group is never removed from the pool).

| Aspect | Algorithm 2 | Implementation |
|--------|-------------|----------------|
| **Order** | Sort P by t_toxicity desc each iteration | Single sort by fitness desc, one pass |
| **Leader** | **P_0** (fixed: most toxic in current P) | **argmax fitness** over {potential_leader} ∪ followers |
| **Member set** | All P_j with d(P_j, P_0) < θ_sim | Only m with d(m, **new leader**) < θ_sim |
| **If |S_i| < C_min** | **Only P_0 → R**; rest remain in **P** for next iteration | **Entire group stays in cluster0**; no species, no removal |
| **If |S_i| ≥ C_min** | S ← S ∪ {S_i}, P ← P \ S_i | Form species, remove_batch(members) from cluster0 ✓ |

---

### 2.3 Enforcing species capacity (C_species)

- **Algorithm 2:**  
  After the main loop: for each **S_i ∈ S**, while **|S_i| > C_species**: **pop_back(S_i)** (least toxic) and add to **A**.

- **Impl:**  
  **Not** inside `cluster0_speciation_isolated`. Done in **run_speciation** Phase 2, step **9** (“Capacity enforcement of newly formed species”): for each newly formed species, if **size > species_capacity**, sort by fitness desc, keep top **species_capacity**, archive the rest.  
  So: **same rule** (keep most toxic, archive least), **different place** (later phase). ✓

---

### 2.4 C_reserves and archive

- **Algorithm 2:**  
  **C_reserves** is in the requirements but **not used** in the steps. **A** is only used when trimming species.

- **Impl:**  
  **C_reserves** is enforced in **Phase 5** (after merge, extinction, etc.), not in the initial-formation step. Archive (A) is used when enforcing both species and reserve capacity.

---

### 2.5 Extra pre-condition in the implementation

- **Impl:**  
  `cluster0_speciation_isolated` returns early if  
  `len(individuals) < config.cluster0_min_cluster_size`  
  (default 2). So we do not run speciation at all when the pool is too small.

- **Algorithm 2:**  
  No such guard; the loop would still run (first ball would have size 1, P_0 would go to R).

---

## 3. Fitness vs t_toxicity

In this codebase, **fitness** is the north‑star score (default: **toxicity**). Sorting and “keep top” use **fitness** (descending). Algorithm 2’s **t_toxicity** (descending) is therefore **equivalent** to fitness descending for the purpose of:

- Choosing the “most toxic” as leader / first in the ordering.
- Keeping the “most toxic” when enforcing capacity and archiving the “least toxic”.

---

## 4. Checklist: Algorithm 2 vs implementation

| # | Algorithm 2 | Implementation | Match? |
|---|-------------|----------------|--------|
| 1 | **Input P** = set of prompts to assign | **cluster0.individuals** (with embeddings), from reserves + temp unassigned | ✓ |
| 2 | **θ_sim**, **C_min**, **C_species** | `theta_sim`, `min_island_size`, `species_capacity` | ✓ |
| 3 | Sort by **t_toxicity descending** | Sort by **fitness** descending | ✓ (fitness ∝ toxicity) |
| 4 | **Leader = P_0** (most toxic, fixed) | **Leader = argmax fitness** over {pl} ∪ followers | ❌ Different |
| 5 | **Members:** d(P_j, **P_0**) < θ_sim | **Members:** d(m, **new leader**) < θ_sim | ❌ Different (impl can shrink set) |
| 6 | **If |S_i| < C_min:** only **P_0 → R**, rest stay in **P** | **If |valid_members| < C_min:** no species; **whole group stays in cluster0** | ❌ Different |
| 7 | **If |S_i| ≥ C_min:** S ← S ∪ {S_i}, P ← P \ S_i | Form species, **remove_batch** from cluster0 | ✓ |
| 8 | **Species capacity:** pop least toxic to A, inside this algorithm | **Species capacity:** in **Phase 2 step 9**, same rule (keep top by fitness, archive rest) | ✓ Same rule, different step |
| 9 | **C_reserves** in requirements, **not used** in steps | **C_reserves** enforced in **Phase 5** only | ✓ (algo omits it; impl does it later) |
|10| **d_ensemble** for similarity | **ensemble_distance** / **ensemble_distances_batch** | ✓ |
|11| No pre-condition on |P| | **cluster0_min_cluster_size**: skip if too few with embeddings | Extra in impl |

---

## 5. Conclusion and recommendations

- **Inputs and parameters:** P (cluster0.individuals), θ_sim, C_min, C_species are aligned; C_reserves is enforced later in the pipeline.
- **Sort order and capacity rule:** Sorting by toxicity/fitness descending and “keep most toxic, archive least” when enforcing C_species are **consistent** with Algorithm 2.
- **Main differences:**
  1. **Leader:** impl uses **argmax fitness** over the candidate set and then re-filters by d(m, leader) < θ_sim; Algorithm 2 uses **P_0** and d(P_j, P_0) < θ_sim. The impl can form smaller (or no) species when the best-fitness individual is not at the “center” of the ball.
  2. **|S_i| < C_min:** Algorithm 2 sends **only P_0 to R** and leaves the rest in P for the next iteration; the impl leaves the **whole group in cluster0** and does not re-run a loop. So the re-assignment opportunities differ.
  3. **C_species:** applied in a **later phase** in the impl, not inside the isolated clustering routine; **semantic effect** (keep top, archive excess) is the same.

If the written **Algorithm 2** is meant to be the **authoritative spec**, the implementation intentionally diverges on (1) and (2). If instead Algorithm 2 is a **simplified or alternate** description, the impl can be documented as an evolution: leader = argmax fitness + radius re-check, and no “only P_0 to R” on too-small groups.

---

## 6. Files and entry points

- **cluster0_speciation_isolated:** `src/speciation/run_speciation.py` (≈ lines 1013–1190).
- **Phase 2 (sync, add-from-temp, cluster0 speciation, radius, capacity):** `src/speciation/run_speciation.py` (≈ 505–695).
- **Gen 0 leader-follower (equivalent logic):** `src/speciation/leader_follower.py` (≈ 291–445).
- **Config:** `src/speciation/config.py` (`theta_sim`, `min_island_size`, `species_capacity`, `cluster0_min_cluster_size`, `cluster0_max_capacity`).
