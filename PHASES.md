# Speciation Pipeline - Phase Overview

## Phase 1: Existing Species Processing
**Functionality:** Process new variants against existing mature species
- Compute embeddings for genomes in temp.json
- Assign variants to closest existing species using leader-follower clustering
- **Radius enforcement:** Clean up members outside radius from updated leaders (Generation N only)
- **Capacity enforcement:** Enforce capacity limits for species that received new members (Generation N)
- Sync cluster 0 with reserves.json

## Phase 2: Cluster 0 Speciation (Isolated)
**Functionality:** Form new species from outliers and reserves
- Load genomes from reserves.json (Cluster 0)
- Apply isolated speciation to form new species from outliers
- Only groups with size >= min_island_size become mature species
- No radius enforcement (Flow 2 requirement)

## Phase 3: Merging + Radius Enforcement
**Functionality:** Merge similar species and enforce radius
- Merge species whose leaders are within theta_merge distance
- Update merged species with highest fitness leader
- **Radius enforcement:** Remove members outside radius from merged species
- After Phase 3, all merged species have members within radius of their leader

## Phase 4: Capacity Enforcement
**Functionality:** Enforce capacity limits for all species
- Enforce capacity limits for ALL species (existing + newly formed + merged)
- Archive excess genomes (keep top species_capacity by fitness)
- Validate no duplicate leader IDs across species
- **After Phase 4, we know:**
  - All members of all species are within radius of their leader (enforced in Phase 1 & 3)
  - All species do not have members exceeding species_capacity (enforced here)

## Phase 5: Freeze & Incubator
**Functionality:** Manage species stagnation and small species
- Track fitness improvements and increment stagnation counters
- Freeze species that haven't improved for species_stagnation generations
- Move species below min_island_size to incubator (Cluster 0)

## Phase 6: Cluster 0 Capacity Enforcement
**Functionality:** Enforce capacity limit for reserves/outliers
- Archive excess genomes from Cluster 0 if size exceeds cluster0_max_capacity
- Keep highest fitness genomes, archive the rest

## Phase 7: Redistribution of Genomes
**Functionality:** Distribute genomes to files based on genome_tracker
- Called within `process_generation()` via `phase8_redistribute_genomes()` function
- Read genome_tracker.json (authoritative source of truth)
- Group genomes by species_id (elites: >0, reserves: 0, archive: -1)
- Write genomes to elites.json, reserves.json, archive.json
- Clear temp.json
- Ensures generation field is set for all distributed genomes (for cumulative metrics)

## Phase 8: Update Metrics & Stats
**Functionality:** Update speciation metrics and statistics
- Update c-TF-IDF labels for all species
- Calculate and log generation summary metrics from distributed files
- Save speciation_state.json with member_ids from elites.json (after Phase 7 distribution)
- Save events_tracker.json, genome_tracker.json
