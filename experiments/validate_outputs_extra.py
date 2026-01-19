#!/usr/bin/env python3
"""Extra validation: duplicate IDs, reserves vs cluster0, species size vs elites, orphans. Stdlib only."""

import json
from pathlib import Path
from collections import defaultdict

def main():
    out = Path("data/outputs/20260118_2218")
    issues, warnings = [], []

    def load(name):
        p = out / name
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    elites = load("elites.json") or []
    reserves = load("reserves.json") or []
    archive = load("archive.json") or []
    temp = load("temp.json") or []
    state = load("speciation_state.json") or {}
    tracker = load("EvolutionTracker.json") or {}

    # --- 1. Duplicate genome IDs across files ---
    seen = {}
    for name, arr in [("elites", elites), ("reserves", reserves), ("archive", archive), ("temp", temp)]:
        if not isinstance(arr, list):
            continue
        for g in arr:
            if not g or not isinstance(g, dict):
                continue
            gid = g.get("id")
            if gid is None:
                continue
            if gid in seen:
                issues.append(f"Duplicate genome id={gid}: in {seen[gid]} and {name}")
            else:
                seen[gid] = name

    # --- 2. Reserves vs cluster0 ---
    reserves_list = [g for g in (reserves or []) if g and isinstance(g, dict)]
    for g in reserves_list:
        if g.get("species_id") is not None and g.get("species_id") != 0:
            issues.append(f"reserves.json: genome id={g.get('id')} has species_id={g.get('species_id')} (expected 0)")
    n_reserves = len(reserves_list)

    c0 = state.get("cluster0") or {}
    c0_size = c0.get("size")
    c0_from_reserves = state.get("cluster0_size_from_reserves")
    if c0_size is not None and n_reserves != c0_size:
        issues.append(f"reserves count={n_reserves} != cluster0.size={c0_size}")
    if c0_from_reserves is not None and n_reserves != c0_from_reserves:
        issues.append(f"reserves count={n_reserves} != cluster0_size_from_reserves={c0_from_reserves}")

    # --- 3. Species: size vs member_ids, and member_ids vs elites ---
    species_dict = state.get("species") or {}
    min_island = (state.get("config") or {}).get("min_island_size", 2)
    elites_by_sid = defaultdict(set)
    for g in elites:
        if not g:
            continue
        sid = g.get("species_id")
        if sid is not None and sid > 0:
            gid = g.get("id")
            if gid is not None:
                elites_by_sid[int(sid)].add(gid)

    for sid, sp in species_dict.items():
        if not isinstance(sp, dict):
            continue
        sid_int = int(sid)
        size = sp.get("size", 0)
        member_ids = sp.get("member_ids") or []
        if size != len(member_ids):
            issues.append(f"Species {sid_int}: size={size} != len(member_ids)={len(member_ids)}")
        elites_n = len(elites_by_sid.get(sid_int, set()))
        if size != elites_n:
            issues.append(f"Species {sid_int}: size={size} != elites count={elites_n} (elites.json species_id={sid_int})")
        if size < min_island and sid_int not in (state.get("incubators") or []):
            # incubator species are tracked by id only; they may have size 0 in state
            warnings.append(f"Species {sid_int}: size={size} < min_island_size={min_island} (not in incubators)")
        for mid in member_ids:
            if mid not in elites_by_sid.get(sid_int, set()):
                # Check if in elites at all with different species_id
                in_elites = any(g.get("id") == mid for g in elites if g)
                if in_elites:
                    issues.append(f"Species {sid_int}: member_id={mid} in member_ids but elites has different species_id")
                else:
                    issues.append(f"Species {sid_int}: member_id={mid} not found in elites.json")

    # --- 4. Orphans: elites with species_id not in state species or incubators ---
    state_ids = set()
    for s in species_dict.keys():
        try:
            state_ids.add(int(s))
        except (ValueError, TypeError):
            pass
    incubator = set(state.get("incubators") or [])
    valid_sid = state_ids | incubator
    for g in elites:
        if not g:
            continue
        sid = g.get("species_id")
        if sid is None or sid <= 0:
            continue
        if int(sid) not in valid_sid:
            issues.append(f"Orphan: elite id={g.get('id')} species_id={sid} not in state species or incubators")

    # --- 5. EvolutionTracker last gen vs files ---
    gens = tracker.get("generations") or []
    if gens:
        last = max(gens, key=lambda x: x.get("generation_number", 0))
        gn = last.get("generation_number")
        et_elites = last.get("elites_count")
        et_reserves = last.get("reserves_count")
        et_arch = last.get("archived_count")
        n_elites = len([g for g in elites if g and g.get("species_id") is not None and g.get("species_id") > 0])
        n_res = len(reserves) if isinstance(reserves, list) else 0
        n_arch = len(archive) if isinstance(archive, list) else 0
        # Elites and reserves are cumulative; EvolutionTracker can store per-gen or cumulative depending on impl.
        # Common: ET stores the count after that generation's distribution. So for last gen, it's the cumulative total.
        if et_elites is not None and et_elites != n_elites:
            warnings.append(f"EvolutionTracker gen {gn} elites_count={et_elites} vs elites.json (species_id>0) count={n_elites}")
        if et_reserves is not None and et_reserves != n_res:
            warnings.append(f"EvolutionTracker gen {gn} reserves_count={et_reserves} vs reserves.json len={n_res}")

    # --- 6. Stagnation ---
    for sid, sp in species_dict.items():
        if isinstance(sp, dict) and sp.get("stagnation", 0) > 0:
            break
    else:
        if species_dict:
            warnings.append("All species have stagnation=0 (expected if parents were always from reserves or parents.json was cleared before speciation; code fix applied for new runs)")

    # --- Report ---
    print("Extra validation: data/outputs/20260118_2218\n")
    print("Issues:", len(issues))
    for i in issues:
        print("  [ISSUE]", i)
    print("Warnings:", len(warnings))
    for w in warnings:
        print("  [WARN]", w)
    if not issues and not warnings:
        print("No extra issues or warnings.")
    return 1 if issues else 0

if __name__ == "__main__":
    exit(main())
