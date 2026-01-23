# Statistics, Metrics, and Output Files Update Analysis

## Summary
This document analyzes when each statistic, metric, and output file field is updated throughout the speciation pipeline to ensure data consistency.

## Event Tracking (_current_gen_events)

**Initialized:** Phase 1 start (line 729)
- `speciation`: 0
- `merge`: 0
- `extinction`: 0
- `moved_to_cluster0`: 0

**Updated:**
- **Phase 1:** `speciation` events (line 807) - when new species form during leader-follower clustering
- **Phase 2:** `speciation` events (line 1276) - when new species form from cluster 0
- **Phase 3:** `merge` events (line 1356) - count of merge events
- **Phase 5:** `extinction` events (line 1790) - count of extinction events
- **Phase 5:** `moved_to_cluster0` events (line 1791) - count of species moved to cluster 0

**Used:**
- **Phase 8:** Metrics calculation (lines 1915-1917) - passed to `record_generation()`

**Status:** ✅ **CORRECT** - Events are tracked throughout pipeline and used in metrics

---

## Archived Count (_archived_count)

**Initialized:** Phase 1 start (line 730)

**Updated:**
- **Throughout:** Incremented in `_archive_individuals()` (line 148) whenever genomes are archived

**Used:**
- **Phase 8:** Result dictionary (line 3121) - returned to caller

**Status:** ✅ **CORRECT** - Counted throughout pipeline

---

## Metrics (metrics_tracker.record_generation)

**Called:** Phase 8 (line 1911)

**Inputs:**
- `generation`: Current generation
- `species`: All species (active + frozen)
- `reserves_size`: From `state["cluster0"].size` (line 1914)
- `speciation_events`: From `_current_gen_events["speciation"]` (line 1915)
- `merge_events`: From `_current_gen_events["merge"]` (line 1916)
- `extinction_events`: From `_current_gen_events["extinction"]` (line 1917)
- `elites_path`: Path to elites.json (line 1919)
- `reserves_path`: Path to reserves.json (line 1920)

**Calculates:**
- Species count (from elites.json)
- Total population (from elites.json + reserves.json)
- Best/avg/std fitness (from elites.json)
- Inter/intra species diversity
- Cluster quality metrics

**Status:** ✅ **CORRECT** - Metrics calculated from distributed files (Phase 7) with correct event counts

---

## speciation_state.json (save_state)

**Called:** Phase 8 (line 1941) - **AFTER** Phase 7 redistribution

**Contains:**
- Species data (active, frozen, extinct, incubator)
- `member_ids`: Read from elites.json (lines 2281-2282, 2304-2305)
- `size`: Count of member_ids (lines 2289, 2311)
- `cluster0.size`: Read from reserves.json (lines 2507-2509)
- Leader data, embeddings, labels, etc.

**Key Fields:**
- **member_ids:** ✅ Read from elites.json (which reflects genome_tracker after Phase 7)
- **size:** ✅ Calculated from member_ids count
- **cluster0.size:** ✅ Read from reserves.json (line 2509)

**Status:** ✅ **CORRECT** - All fields updated from distributed files

**Note:** `_update_speciation_state_cluster0_size_after_distribution()` is called in Phase 7 (line 1880), but `save_state()` is called in Phase 8 (line 1941). This means:
1. Phase 7: Distribution happens → reserves.json created
2. Phase 7: cluster0.size updated in speciation_state.json (line 1880)
3. Phase 8: save_state() called → reads cluster0.size from reserves.json again (line 2509) → overwrites with same value

**Potential Issue:** ⚠️ **REDUNDANT** - cluster0.size is updated twice, but both use same source (reserves.json), so it's correct but inefficient.

---

## genome_tracker.json

**Updated:** Throughout all phases
- Phase 1: When variants assigned to species
- Phase 2: When new species formed
- Phase 3: When species merged
- Phase 4: When genomes archived (capacity enforcement)
- Phase 5: When species moved to cluster 0
- Phase 6: When cluster 0 capacity enforced

**Saved:** Phase 8 (line 1952)

**Status:** ✅ **CORRECT** - Tracker updated throughout, saved at end

---

## events_tracker.json

**Updated:** Throughout all phases
- Events logged: `clustering_assigned`, `capacity_archived`, `species_merged`, `reassigned_from_archive`, etc.

**Saved:** Phase 8 (line 1946)

**Status:** ✅ **CORRECT** - Events logged throughout, saved at end

---

## elites.json, reserves.json, archive.json

**Written:** Phase 7 (via `phase8_redistribute_genomes()`)

**Source:** genome_tracker.json (authoritative source of truth)

**Contains:**
- Full genome data with `species_id` matching tracker
- Genomes grouped by `species_id`:
  - `species_id > 0` → elites.json
  - `species_id == 0` → reserves.json
  - `species_id == -1` → archive.json

**Status:** ✅ **CORRECT** - Files written based on tracker state

---

## cluster0.size (in-memory)

**Updated:**
- Throughout phases when genomes added/removed from cluster 0
- Phase 6: After capacity enforcement

**Synced with reserves.json:**
- Phase 7: After distribution, `_update_speciation_state_cluster0_size_after_distribution()` updates speciation_state.json
- Phase 8: `save_state()` reads from reserves.json (line 2509) and validates against in-memory size (line 2513)

**Status:** ✅ **CORRECT** - Size tracked in-memory and synced with file

---

## Potential Issues Found

### 1. Redundant cluster0.size Update
**Location:** Phase 7 (line 1880) and Phase 8 (line 2509)
**Issue:** cluster0.size is updated twice:
- Phase 7: `_update_speciation_state_cluster0_size_after_distribution()` updates it
- Phase 8: `save_state()` reads from reserves.json and updates it again

**Impact:** Low - Both use same source (reserves.json), so result is correct but redundant
**Recommendation:** Remove redundant update in Phase 7 since `save_state()` handles it

### 2. Metrics Use cluster0.size from Memory
**Location:** Phase 8, line 1914
**Issue:** `reserves_size` passed to metrics uses `state["cluster0"].size` (in-memory) instead of reading from reserves.json

**Impact:** Low - Should match, but could be inconsistent if in-memory state is stale
**Recommendation:** Read from reserves.json for accuracy (like `save_state()` does)

---

## Overall Assessment

✅ **MOSTLY CORRECT** - All statistics, metrics, and file fields are updated at appropriate times:
- Events tracked throughout pipeline ✅
- Metrics calculated from distributed files ✅
- speciation_state.json reads member_ids from elites.json ✅
- genome_tracker.json updated throughout and saved ✅
- Output files (elites.json, reserves.json, archive.json) written in Phase 7 ✅

⚠️ **MINOR ISSUES:**
- Redundant cluster0.size update (cosmetic, doesn't affect correctness)
- Metrics uses in-memory cluster0.size instead of file (should match, but file is more authoritative)
