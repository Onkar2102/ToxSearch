# Algorithm Validation: Leader-Follower Clustering vs Implementation

## Summary

The LaTeX algorithm **partially reflects** the implementation. Several steps match, but there are important differences in **structure** (where Step 1b is executed), **parameter mapping**, **one typo/bug** in the algorithm, and **missing details** (stagnation increment, merge post-verification, processing order). Below: parameter map, step-by-step comparison, and a **corrected algorithm** that matches the code.

---

## 1. Parameter Mapping

| Algorithm | Implementation | Notes |
|-----------|----------------|-------|
| $P$ | `temp.json` genomes (with `prompt_embedding`) | ✓ |
| $\mathcal{S}$ | `state["species"]` | ✓ |
| $R$ | `state["cluster0"]` / `reserves.json` | ✓ |
| $\theta_{\mathrm{sim}}$ | `config.theta_sim` | ✓ |
| $\theta_{\mathrm{merge}}$ | `config.theta_merge` | ✓ |
| $C_{\min}$ | `config.min_island_size` | ✓ |
| $C_{\mathrm{species}}$ | `config.species_capacity` | ✓ |
| $C_{\mathrm{reserves}}$ | `config.cluster0_max_capacity` | ✓ |
| $T_{\mathrm{species}}$ | `config.species_stagnation` | ✓ |
| $T_{\mathrm{merge}}$ | `min_stability_gens` in `process_merges` (default **1**) | Not in `SpeciationConfig`; fixed in code |

---

## 2. Step-by-Step Comparison

### Step 1: Assign variants to species or reserves

- **1a. Find nearest $S_i$ with $d_{\mathrm{ensemble}}(p, \mathrm{leader}(S_i)) < \theta_{\mathrm{sim}}$**  
  - **Impl:** `leader_follower_clustering` checks existing species leaders; assigns to **closest** if `min_dist < theta_sim`. ✓

- **1a. Add $p$ to $S_i$; if $\hat{f}(p) > \hat{f}(\mathrm{leader}(S_i))$ then $\mathrm{leader}(S_i) \gets p$, $\mathrm{stagnation}(S_i) \gets 0$**  
  - **Impl:** Adds to species; if `ind.fitness > sp.leader.fitness` then leader is updated. **Stagnation reset:** `stagnation = 0` only when `ind.fitness > sp.max_fitness` (not only when $> \mathrm{leader}$). So the **stagnation reset condition is stricter** in the impl (improvement over `max_fitness`).

- **1b. For each $r \in R$, if $d_{\mathrm{ensemble}}(p,r) < \theta_{\mathrm{sim}}$: add to $\mathrm{followers}(r)$; if $|\{r\} \cup \mathrm{followers}(r)| \geq C_{\min}$, form $S_{\mathrm{new}}$, remove from $R$**  
  - **Impl:** In the **main pipeline**, `leader_follower_clustering` is called with **`skip_cluster0_outliers=True`**, so **Step 1b is not executed** during variant assignment.  
  - **Formation from $R$** is done in **Phase 2** by `cluster0_speciation_isolated`, which runs on `cluster0.individuals` (reserves + unassigned from temp) in a **separate** step: potential leaders in $R$ gain followers; when $|\mathrm{leader} \cup \mathrm{followers}| \geq \texttt{min\_island\_size}$, a new species is created and those individuals are removed from cluster0 via `remove_batch`.  
  - **Structural difference:** The algorithm embeds 1b in the same loop as 1a/1c; the implementation splits: (1a, 1c) in `leader_follower_clustering`, and the “1b-like” logic in `cluster0_speciation_isolated`.

- **1c. $R \gets p$**  
  - **Typo:** Should be $R \gets R \cup \{p\}$ (add $p$ to $R$). Impl: `ind.species_id = CLUSTER_0_ID`; individual is later in cluster0/reserves. ✓ (with notation fix)

- **Bug in algorithm (line with $S_{\mathrm{new}}$):**  
  - **Current:** $S_{\mathrm{new}} \gets |\{r\} \cup \mathrm{followers}(r)|$ (assigns **cardinality**).  
  - **Should be:** Form new species $S_{\mathrm{new}}$ from $\{r\} \cup \mathrm{followers}(r)$; $\mathcal{S} \gets \mathcal{S} \cup \{S_{\mathrm{new}}\}$; remove $r$ and assigned followers from $R$.

- **Processing order:** Impl **sorts** $P$ by fitness (descending) and processes in that order; algorithm does not specify. Impl also picks **leader** of a new species as $\arg\max \hat{f}$ over $\{r\} \cup \mathrm{followers}(r)$ and then **filters** members to those within $\theta_{\mathrm{sim}}$ of that leader before creating the species; algorithm does not state this.

---

### Step 2: Radius cleanup

- **Remove $m$ with $d_{\mathrm{ensemble}}(m, \mathrm{leader}(S_i)) \geq \theta_{\mathrm{sim}}$**  
  - **Impl:** `run_speciation` removes members with `dist >= theta_sim` and moves them to cluster0. ✓ (impl uses $\geq$)

- **Scope:** Algorithm says “for each $S_i \in \mathcal{S}$”. Impl applies radius cleanup only to species that **received new members** in Phase 1, plus **newly formed species** in Phase 2. For species with no new members, distances are unchanged, so this is an optimization; semantically consistent.

---

### Step 3: Enforce capacities

- **Species: if $|S_i| > C_{\mathrm{species}}$, sort by $t_{\mathrm{toxicity}}$, keep top $C_{\mathrm{species}}$, archive excess**  
  - **Impl:** Sort by `fitness` (north-star, e.g. toxicity), keep top `species_capacity`, archive excess. ✓

- **Reserves: if $|R| > C_{\mathrm{reserves}}$, sort by $t_{\mathrm{toxicity}}$, keep top $C_{\mathrm{reserves}}$, archive excess**  
  - **Impl:** Reserves (cluster0) capacity is enforced in **Phase 5**, **after** merging (Step 4) and deactivation (Step 5). So the **order** differs: algorithm does $R$ capacity right after species capacity in a single “Step 3”; implementation does cluster0 capacity once at the end. Effect is the same (both cap $R$).

---

### Step 4: Merging

- **If two leaders within $\theta_{\mathrm{merge}}$, combine species**  
  - **Impl:** `process_merges` finds pairs with `d(leader_i, leader_j) < theta_merge` and merges them. ✓

- **$T_{\mathrm{merge}}$ (min. generations to be mergeable):**  
  - **Impl:** `min_stability_gens` in `process_merges`; stability: $(G - \mathrm{created\_at}) \geq \mathrm{min\_stability\_gens}$. Default 1; not passed from `run_speciation`, so $T_{\mathrm{merge}} = 1$ in practice.

- **Post-merge radius check:**  
  - **Impl:** `merge_islands` builds the merged species, then checks each member against the **new** leader; those with $d \geq \theta_{\mathrm{sim}}$ are **outliers** and are moved to cluster0 by the caller. The algorithm does **not** mention this.

---

### Step 5: Deactivation

- **$\mathrm{stagnation}(S_i) \geq T_{\mathrm{species}} \Rightarrow \mathrm{species\_state}(S_i) \gets \mathrm{frozen}$**  
  - **Impl:** `process_extinctions`: `stagnation >= species_stagnation` $\Rightarrow$ `species_state = "frozen"`. ✓  
  - **Stagnation rule:** Impl: `record_fitness(was_selected_as_parent, max_fitness_increased)`. If `max_fitness_increased` then `stagnation = 0`; else if `was_selected_as_parent` then `stagnation += 1`; else unchanged. The algorithm only describes the **reset** when $f(p) > f(\mathrm{leader})$ during assignment; it does **not** describe when stagnation is incremented (selected as parent, no improvement).

- **$|S_i| < C_{\min} \Rightarrow R \gets S_i$**  
  - **Impl:** `process_extinctions`: species with `size < min_size` have all members moved to cluster0; species get `species_state = "incubator"`, are removed from the active species dict, and stored in `historical_species`. The algorithm’s “$R \gets S_i$” is underspecified; impl is consistent (members go to $R$).

- **If $|R| > C_{\mathrm{reserves}}$ again**  
  - **Impl:** Phase 5 enforces cluster0 capacity after `process_extinctions`, so $R$ is capped again. ✓

---

## 3. Corrected Algorithm (Matches Implementation)

Below is a version that aligns with the code: separate **assignment** (1a, 1c) from **cluster-0 speciation** (1b), fix the $S_{\mathrm{new}}$ line, add merge post-verification and stagnation rules, and respect the actual order of reserve capacity.

```
Step 1: Assign variants to existing species or reserves
  - Sort P by fitness (descending).
  - For each p in P:
    1a. Find nearest S_i in S with d_ensemble(p, leader(S_i)) < θ_sim.
        If found: add p to S_i. If f(p) > f(leader(S_i)): leader(S_i) <- p.
                  If f(p) > max_fitness(S_i): stagnation(S_i) <- 0.
        Continue to next p.
    1c. R <- R ∪ {p}.

Step 1b (Cluster 0 speciation, separate phase):
  - Set cluster0 = reserves R plus unassigned from P with species_id=0 (and with embeddings).
  - On cluster0: sort by fitness (desc). Treat individuals as potential leaders with followers.
  - For each unassigned in that order:
        If d(p, r) < θ_sim for some potential leader r: add p to followers(r).
        If |{r} ∪ followers(r)| >= C_min:
            Let leader = argmax_{m in {r}∪followers(r)} f(m).
            Keep only m with d(m, leader) < θ_sim; if that set has size >= C_min:
                Form S_new from that set; S <- S ∪ {S_new}; remove them from R (and from cluster0).

Step 2: Radius cleanup
  - For each S_i that received new members (and for each newly formed species):
        Remove members m with d_ensemble(m, leader(S_i)) >= θ_sim; move them to R.

Step 3: Species capacity
  - For each S_i: if |S_i| > C_species, sort by fitness, keep top C_species, archive excess.

Step 4: Merging
  - Pairs (S_i, S_j) with d(leader_i, leader_j) < θ_merge and (G - created_at) >= T_merge for both:
        Merge into new species; pick new leader = argmax fitness over combined members;
        keep only members with d(m, new_leader) < θ_sim; rest are outliers -> R.
        Parents removed from S; merged species added.

Step 5: Deactivation
  - Stagnation: for each species, record_fitness(was_selected, max_fitness_increased).
    If max_fitness_increased: stagnation <- 0. Else if was_selected: stagnation += 1.
  - If stagnation(S_i) >= T_species: species_state(S_i) <- "frozen".
  - If |S_i| < C_min: move all members to R; remove S_i from S (incubator).
  - If |R| > C_reserves: sort by fitness, keep top C_reserves, archive excess.
```

---

## 4. Checklist of Fixes for the LaTeX Algorithm

1. **1c:** Use $R \gets R \cup \{p\}$ (or “add $p$ to $R$”) instead of $R \gets p$.
2. **1b line:** Replace  
   `S_new <- |{r} ∪ followers(r)|`  
   by: form new species $S_{\mathrm{new}}$ from $\{r\} \cup \mathrm{followers}(r)$ (with leader = $\arg\max \hat{f}$ and optionally a radius filter); $\mathcal{S} \gets \mathcal{S} \cup \{S_{\mathrm{new}}\}$; remove $r$ and assigned followers from $R$.
3. **Structure:** Either (a) clearly separate “Step 1: assign to $\mathcal{S}$ or $R$” from “Step 1b: cluster-0 speciation on $R$ (and temp unassigned)”, or (b) add a short note that the implementation performs 1b in a distinct phase (cluster0 speciation) after 1a/1c.
4. **Stagnation:**  
   - In 1a: “if $\hat{f}(p) > \hat{f}(\mathrm{leader}(S_i))$: $\mathrm{leader}(S_i) \gets p$; if $\hat{f}(p) > \mathrm{max\_fitness}(S_i)$ then $\mathrm{stagnation}(S_i) \gets 0$.”  
   - Add a “Stagnation update” before Step 5: “For each $S_i$: if $\mathrm{max\_fitness}(S_i)$ increased this generation then $\mathrm{stagnation}(S_i) \gets 0$; else if $S_i$ was selected as parent then $\mathrm{stagnation}(S_i) \gets \mathrm{stagnation}(S_i) + 1$.”
5. **Merge:** Add: “After merging, keep only members with $d(m, \mathrm{new\_leader}) < \theta_{\mathrm{sim}}$; move the rest to $R$.”
6. **$R$ capacity:** Note that $|R| > C_{\mathrm{reserves}}$ is enforced both after Step 3 (in some flows) and again after Step 5 (deactivation); the implementation does the final reserves cap after Step 5.
7. **$T_{\mathrm{merge}}$:** Define as “min. generations since $\mathrm{created\_at}$ for a species to be mergeable” and state that it is 1 in the current implementation (min_stability_gens).
8. **Processing order:** Optionally state that $P$ is processed in order of decreasing $\hat{f}$ (fitness).

---

## 5. Files Reference

- **Step 1 (1a, 1c):** `src/speciation/leader_follower.py` (with `skip_cluster0_outliers=True` in `run_speciation`).
- **Step 1b:** `src/speciation/run_speciation.py` (Phase 2: add temp unassigned to cluster0, then `cluster0_speciation_isolated`); `cluster0_speciation_isolated` in `run_speciation.py` and `reserves.py` (logic).
- **Steps 2–3, 5 (R capacity):** `src/speciation/run_speciation.py` (Phases 1, 2, 5).
- **Step 4:** `src/speciation/merging.py` (`process_merges`, `merge_islands`).
- **Step 5 (freeze, small→R):** `src/speciation/extinction.py` (`process_extinctions`).
- **Stagnation:** `src/speciation/species.py` (`record_fitness`); `run_speciation.py` Phase 4 (record_fitness, then process_extinctions).
- **Config:** `src/speciation/config.py`; `merging.process_merges` for `min_stability_gens`.
