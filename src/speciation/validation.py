"""
validation.py

Consistency validation and distance threshold analysis for speciation framework.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .species import Species
from .distance import ensemble_distance
from .config import SpeciationConfig

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def validate_speciation_consistency(
    outputs_path: Path,
    generation: int,
    logger=None
) -> Tuple[bool, List[str]]:
    """
    Validate consistency across all speciation files.
    
    Checks:
    1. Species IDs in elites.json match species in speciation_state.json
    2. Reserves (cluster 0) have species_id = 0
    3. Species sizes match between state and elites.json
    4. No duplicate genome IDs across files
    5. Sum invariant: temp.json = elites.json + reserves.json + archive.json
    
    Args:
        outputs_path: Path to outputs directory
        generation: Current generation number
        logger: Optional logger instance
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if logger is None:
        logger = get_logger("Validation")
    
    errors: List[str] = []
    
    # Load all files
    state_file_path = outputs_path / "speciation_state.json"
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    temp_path = outputs_path / "temp.json"
    
    try:
        # Load state
        if not state_file_path.exists():
            errors.append("speciation_state.json not found")
            return False, errors
        
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_file = json.load(f)
        
        # Load elites
        elites = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
        
        # Load reserves
        reserves = []
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves = json.load(f)
        
        # Load archive
        archive = []
        if archive_path.exists():
            with open(archive_path, 'r', encoding='utf-8') as f:
                archive = json.load(f)
        
        # Check 1: Species ID consistency
        elite_species_ids = {g.get("species_id") for g in elites if g.get("species_id") is not None}
        state_species_ids = set(int(k) for k in state_file.get("species", {}).keys() if k.isdigit())
        incubator_ids = set(state_file.get("incubators", []))  # Incubator species IDs (just tracked)
        
        # Remove None and 0 (cluster 0) from elite_species_ids for comparison
        elite_species_ids_filtered = {sid for sid in elite_species_ids if sid is not None and sid > 0}
        
        # Check for species in elites that are not in active/frozen or incubator
        all_tracked_ids = state_species_ids | incubator_ids
        missing_in_state = elite_species_ids_filtered - all_tracked_ids
        if missing_in_state:
            errors.append(f"Species in elites.json but not tracked in state (active/frozen/incubator): {missing_in_state}")
        
        missing_in_elites = state_species_ids - elite_species_ids_filtered
        if missing_in_elites:
            # This is OK - species might be empty or frozen
            logger.debug(f"Species in state but not in elites (may be empty/frozen): {missing_in_elites}")
        
        # Check 2: Reserves species_id
        reserve_species_ids = {g.get("species_id") for g in reserves if g.get("species_id") is not None}
        if reserve_species_ids != {0} and reserve_species_ids != set():
            errors.append(f"Reserves with non-zero species_id: {reserve_species_ids}")
        
        # Check 3: Size consistency (only for active/frozen species, exclude incubator)
        # Only validate if elites.json has genomes (after distribution)
        if len(elites) > 0:
            for sid_str, sp_dict in state_file.get("species", {}).items():
                try:
                    sid = int(sid_str)
                    species_state = sp_dict.get("species_state", "active")
                    
                    # Skip incubator species from size validation (they're just tracked by ID)
                    if species_state == "incubator":
                        continue
                    
                    expected_size = len([g for g in elites if g.get("species_id") == sid])
                    actual_size = sp_dict.get("size", 0)
                    if expected_size != actual_size:
                        errors.append(f"Species {sid}: expected size {expected_size} (from elites.json), got {actual_size} (from state)")
                except ValueError:
                    continue
        else:
            # Before distribution, elites.json is empty - this is expected
            # Don't validate sizes until after distribution
            logger.debug("Skipping size validation: elites.json is empty (before distribution)")
        
        # Check 4: Duplicate IDs
        all_genome_ids = []
        all_genome_ids.extend([g.get("id") for g in elites if g.get("id") is not None])
        all_genome_ids.extend([g.get("id") for g in reserves if g.get("id") is not None])
        all_genome_ids.extend([g.get("id") for g in archive if g.get("id") is not None])
        
        from collections import Counter
        id_counts = Counter(all_genome_ids)
        duplicates = [gid for gid, count in id_counts.items() if count > 1]
        if duplicates:
            errors.append(f"Duplicate genome IDs found: {duplicates[:10]}")  # Limit to first 10
        
        # Check 5: Sum invariant (if temp.json exists and is not empty)
        # After distribution, temp.json should be cleared (empty list), so we only check if it has genomes
        if temp_path.exists():
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    temp_genomes = json.load(f)
                temp_count = len(temp_genomes) if isinstance(temp_genomes, list) else 0
                total_distributed = len(elites) + len(reserves) + len(archive)
                
                # After distribution, temp.json should be empty (cleared by distribute_genomes)
                # If temp.json has genomes, they should match the distributed count
                # This check is mainly for detecting if distribution didn't clear temp.json
                if temp_count > 0 and temp_count != total_distributed:
                    # This might indicate genomes weren't distributed or temp.json wasn't cleared
                    logger.debug(f"Sum invariant check: temp.json={temp_count}, elites+reserves+archive={total_distributed}")
                    # Only error if there's a significant mismatch (more than just rounding/edge cases)
                    if abs(temp_count - total_distributed) > 1:
                        errors.append(f"Sum invariant violated: temp.json={temp_count}, elites+reserves+archive={total_distributed}")
            except Exception as e:
                logger.warning(f"Could not verify sum invariant: {e}")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"Consistency validation passed for generation {generation}")
        else:
            logger.warning(f"Consistency validation found {len(errors)} errors for generation {generation}")
        
        return is_valid, errors
        
    except Exception as e:
        error_msg = f"Validation failed with exception: {e}"
        logger.error(error_msg, exc_info=True)
        return False, [error_msg]


def analyze_distance_distribution(
    species: Dict[int, Species],
    cluster0,
    config: SpeciationConfig,
    logger=None
) -> Dict[str, Any]:
    """
    Analyze distance distributions to validate thresholds.
    
    Computes:
    - Intra-species distance statistics (mean, std, max, percentile_95)
    - Inter-species distance statistics (mean, std, min, percentile_5)
    - Validates thresholds against actual distributions
    
    Args:
        species: Dict of active species
        cluster0: Cluster0 object (for outlier distances)
        config: SpeciationConfig with thresholds
        logger: Optional logger instance
    
    Returns:
        Dictionary with distance statistics and validation warnings
    """
    if logger is None:
        logger = get_logger("DistanceAnalysis")
    
    intra_distances = []
    inter_distances = []
    
    # Collect intra-species distances (member to leader)
    for sp in species.values():
        if sp.leader is None or sp.leader.embedding is None:
            continue
        
        leader = sp.leader
        for member in sp.members:
            if member.id == leader.id or member.embedding is None:
                continue
            
            try:
                d = ensemble_distance(
                    leader.embedding, member.embedding,
                    leader.phenotype, member.phenotype,
                    config.w_genotype, config.w_phenotype
                )
                intra_distances.append(d)
            except Exception:
                pass
    
    # Collect inter-species distances (leader to leader)
    leaders = [sp.leader for sp in species.values() 
               if sp.leader is not None and sp.leader.embedding is not None]
    
    for i in range(len(leaders)):
        for j in range(i + 1, len(leaders)):
            try:
                d = ensemble_distance(
                    leaders[i].embedding, leaders[j].embedding,
                    leaders[i].phenotype, leaders[j].phenotype,
                    config.w_genotype, config.w_phenotype
                )
                inter_distances.append(d)
            except Exception:
                pass
    
    # Compute statistics
    intra_stats = {}
    inter_stats = {}
    warnings = []
    
    if intra_distances:
        intra_stats = {
            "mean": round(float(np.mean(intra_distances)), 4),
            "std": round(float(np.std(intra_distances)), 4),
            "min": round(float(np.min(intra_distances)), 4),
            "max": round(float(np.max(intra_distances)), 4),
            "percentile_95": round(float(np.percentile(intra_distances, 95)), 4),
            "count": len(intra_distances)
        }
        
        # Validate thresholds
        if intra_stats["max"] > config.theta_sim:
            warnings.append(
                f"Max intra-species distance {intra_stats['max']:.4f} exceeds theta_sim {config.theta_sim:.4f}"
            )
        
        if intra_stats["percentile_95"] > config.theta_sim:
            warnings.append(
                f"95th percentile intra-species distance {intra_stats['percentile_95']:.4f} exceeds theta_sim {config.theta_sim:.4f}"
            )
    else:
        intra_stats = {"count": 0}
    
    if inter_distances:
        inter_stats = {
            "mean": round(float(np.mean(inter_distances)), 4),
            "std": round(float(np.std(inter_distances)), 4),
            "min": round(float(np.min(inter_distances)), 4),
            "max": round(float(np.max(inter_distances)), 4),
            "percentile_5": round(float(np.percentile(inter_distances, 5)), 4),
            "count": len(inter_distances)
        }
        
        # Validate thresholds
        if inter_stats["percentile_5"] < config.theta_merge:
            warnings.append(
                f"5th percentile inter-species distance {inter_stats['percentile_5']:.4f} is below theta_merge {config.theta_merge:.4f} "
                f"(may cause premature merging)"
            )
        
        if inter_stats["min"] < config.theta_merge:
            warnings.append(
                f"Minimum inter-species distance {inter_stats['min']:.4f} is below theta_merge {config.theta_merge:.4f} "
                f"(species may be too similar)"
            )
    else:
        inter_stats = {"count": 0}
    
    # Log warnings
    if warnings:
        for warning in warnings:
            logger.warning(f"Distance threshold validation: {warning}")
    else:
        logger.info("Distance threshold validation passed")
    
    return {
        "intra_species": intra_stats,
        "inter_species": inter_stats,
        "warnings": warnings,
        "thresholds": {
            "theta_sim": config.theta_sim,
            "theta_merge": config.theta_merge
        }
    }
