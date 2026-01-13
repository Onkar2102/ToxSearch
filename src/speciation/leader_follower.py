"""
leader_follower.py

Leader-Follower clustering algorithm for speciation.
Reads temp.json and speciation_state.json directly, performs clustering,
and updates both files directly.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, TYPE_CHECKING
from pathlib import Path

from .species import Individual, Species, generate_species_id, SpeciesIdGenerator
from .distance import ensemble_distance, ensemble_distances_batch
from .reserves import CLUSTER_0_ID

if TYPE_CHECKING:
    from .reserves import Cluster0

from utils import get_custom_logging
from utils import get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _ = get_system_utils()


def leader_follower_clustering(
    temp_path: Optional[str] = None,
    speciation_state_path: Optional[str] = None,
    theta_sim: float = 0.2,
    current_generation: int = 0,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    logger=None) -> Tuple[Dict[int, Species], Set[int]]:
    """
    Leader-Follower clustering algorithm that reads and writes files directly.
    
    This function uses DEFERRED SPECIES ID ASSIGNMENT:
    - Species IDs are only assigned when a leader gains at least one follower
    - Individuals that don't fit existing species become "potential leaders"
    - Potential leaders become actual species only when another individual joins them
    - Potential leaders without followers stay in cluster 0 (reserves)
    
    This ensures:
    - No wasted species IDs
    - No single-member species to clean up
    - Species inherently have minimum size of 2
    
    Pipeline:
    1. Reads genomes from temp.json (must have prompt_embedding field)
    2. Reads existing species from speciation_state.json (if exists)
    3. For Generation 0 (no species exist):
       - First individual becomes first potential leader
       - Subsequent individuals check against potential leaders
       - If within theta_sim of a potential leader, species forms (ID assigned)
       - If not, individual becomes a new potential leader
       - Potential leaders without followers go to cluster 0
    4. For Generation N:
       - Checks each genome against existing species leaders
       - If within theta_sim, assigns to that species
       - If not, checks against cluster 0 outliers (from reserves.json) as potential leaders
       - If within theta_sim of an outlier, forms a new species (outlier + new individual)
       - If not within radius of any leader or outlier, adds to cluster 0
       - check_speciation() handles additional species formation from remaining cluster 0 individuals
    5. Updates temp.json with species_id for each genome
    6. Updates speciation_state.json with new species structure
    
    Args:
        temp_path: Path to temp.json (defaults to outputs_path / "temp.json")
        speciation_state_path: Path to speciation_state.json (defaults to outputs_path / "speciation_state.json")
        theta_sim: Semantic distance threshold for species assignment
        current_generation: Current generation number
        logger: Optional logger instance
    
    Returns:
        Tuple of (Dict mapping species_id -> Species, Set of species_ids that received new members)
    """
    if logger is None:
        logger = get_logger("LeaderFollowerClustering")
    
    # Track which species received new members (for optimization)
    species_with_new_members = set()
    
    # Determine file paths
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    if speciation_state_path is None:
        outputs_path = get_outputs_path()
        speciation_state_path = str(outputs_path / "speciation_state.json")
    
    temp_path_obj = Path(temp_path)
    speciation_state_path_obj = Path(speciation_state_path)
    
    if not temp_path_obj.exists():
        logger.error(f"Temp file not found: {temp_path}")
        return {}
    
    # Read genomes from temp.json
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    if not genomes:
        logger.warning("No genomes found in temp.json")
        return {}
    
    # Convert genomes to Individual objects
    population = [Individual.from_genome(genome) for genome in genomes]
    
    # Filter out individuals without embeddings
    valid_population = [ind for ind in population if ind.embedding is not None]
    if not valid_population:
        logger.error("No individuals with embeddings")
        return {}
    
    # Read existing species from speciation_state.json (if exists)
    existing_species: Dict[int, Species] = {}
    
    if speciation_state_path_obj.exists():
        try:
            with open(speciation_state_path_obj, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore species (excluding species 0)
            for sid_str, sp_dict in state.get("species", {}).items():
                sid = int(sid_str)
                if sid == CLUSTER_0_ID:
                    continue  # Skip species 0 (it's cluster0/reserves)
                
                # Reconstruct leader Individual
                leader_embedding = None
                if sp_dict.get("leader_embedding"):
                    leader_embedding = np.array(sp_dict["leader_embedding"])
                
                # Extract phenotype if available in genome_data
                leader_phenotype = None
                if sp_dict.get("leader_genome_data"):
                    from .phenotype_distance import extract_phenotype_vector
                    leader_phenotype = extract_phenotype_vector(sp_dict["leader_genome_data"], logger=logger)
                
                leader = Individual(
                    id=sp_dict["leader_id"],
                    prompt=sp_dict.get("leader_prompt", ""),
                    fitness=sp_dict.get("leader_fitness", 0.0),
                    embedding=leader_embedding,
                    phenotype=leader_phenotype,
                    species_id=sid
                )
                
                # Reconstruct species (members will be populated during clustering)
                species = Species(
                    id=sid,
                    leader=leader,
                    members=[leader],
                    radius=sp_dict.get("radius", theta_sim),
                    stagnation=sp_dict.get("stagnation", 0),
                    max_fitness=sp_dict.get("max_fitness", leader.fitness),
                    species_state=sp_dict.get("species_state", "active"),
                    created_at=sp_dict.get("created_at", 0),
                    last_improvement=sp_dict.get("last_improvement", 0),
                    fitness_history=sp_dict.get("fitness_history", []),
                    labels=sp_dict.get("labels", []),
                    label_history=sp_dict.get("label_history", []),
                    cluster_origin=sp_dict.get("cluster_origin", "natural"),  # Default to "natural" if None
                    parent_ids=sp_dict.get("parent_ids"),
                    leader_distance=sp_dict.get("leader_distance", 0.0)
                )
                existing_species[sid] = species
            
            # Update SpeciesIdGenerator to avoid ID conflicts
            if existing_species:
                max_species_id = max(existing_species.keys())
                SpeciesIdGenerator.set_min_id(max_species_id + 1)
            
            logger.info(f"Loaded {len(existing_species)} existing species from speciation_state.json")
            
        except Exception as e:
            logger.warning(f"Failed to load speciation_state.json: {e}, starting fresh")
            existing_species = {}
    
    # Determine if this is Generation 0 (no species exist, or only species 0)
    is_generation_0 = len(existing_species) == 0
    
    # Load cluster 0 outliers for Generation N (if not already loaded above)
    cluster0_outliers: List[Tuple[np.ndarray, Optional[np.ndarray], Individual]] = []
    if not is_generation_0:
        outputs_path = get_outputs_path()
        reserves_path = outputs_path / "reserves.json"
        if reserves_path.exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                for genome in reserves_genomes:
                    if genome.get("prompt_embedding"):
                        outlier_emb = np.array(genome["prompt_embedding"])
                        # Extract phenotype if available
                        outlier_pheno = None
                        if genome.get("moderation_result"):
                            from .phenotype_distance import extract_phenotype_vector
                            outlier_pheno = extract_phenotype_vector(genome.get("moderation_result"), logger=logger)
                        outlier_ind = Individual.from_genome(genome)
                        if outlier_ind.embedding is not None:
                            cluster0_outliers.append((outlier_emb, outlier_pheno, outlier_ind))
                logger.debug(f"Loaded {len(cluster0_outliers)} cluster 0 outliers from reserves.json")
            except Exception as e:
                logger.warning(f"Failed to load cluster 0 outliers from reserves.json: {e}")
    
    # Sort population by fitness (descending) - highest fitness processed first
    sorted_pop = sorted(valid_population, key=lambda x: x.fitness, reverse=True)
    
    # Initialize species dict
    species: Dict[int, Species] = existing_species.copy() if existing_species else {}
    leaders: List[Tuple[int, np.ndarray, Optional[np.ndarray]]] = [
        (sid, sp.leader.embedding, sp.leader.phenotype) 
        for sid, sp in species.items() 
        if sp.leader.embedding is not None
    ]
    
    # DEFERRED SPECIES ID ASSIGNMENT
    # For Generation 0: Use potential leaders - species only form when a follower appears
    # For Generation N: Use existing species leaders
    
    # Potential leaders: List of (species_id_or_None, embedding, phenotype, Individual)
    # species_id is None until a follower is found, then it's assigned
    potential_leaders: List[Tuple[Optional[int], np.ndarray, Optional[np.ndarray], Individual]] = []
    
    # For Generation N: Track cluster 0 outliers as potential leaders
    # Format: (None, embedding, phenotype, Individual) - None means no species yet
    cluster0_potential_leaders: List[Tuple[Optional[int], np.ndarray, Optional[np.ndarray], Individual]] = []
    if not is_generation_0 and cluster0_outliers:
        # Convert cluster 0 outliers to potential leaders
        for outlier_emb, outlier_pheno, outlier_ind in cluster0_outliers:
            cluster0_potential_leaders.append((None, outlier_emb, outlier_pheno, outlier_ind))
        logger.debug(f"Generation N: Loaded {len(cluster0_potential_leaders)} cluster 0 outliers as potential leaders")
    
    if is_generation_0:
        # First individual becomes first potential leader (no species yet)
        first = sorted_pop[0]
        first.species_id = CLUSTER_0_ID  # Stays in cluster 0 until follower found
        potential_leaders.append((None, first.embedding, first.phenotype, first))
        remaining_pop = sorted_pop[1:]
        logger.debug(f"Generation 0: First individual {first.id} becomes potential leader (cluster 0)")
    else:
        # Generation N: process all individuals against existing species and cluster 0 outliers
        remaining_pop = sorted_pop
    
    # Process remaining individuals
    for ind in remaining_pop:
        assigned = False
        min_dist = float('inf')
        nearest_leader_id = None
        
        # 1. First check against existing species leaders (Gen N, or Gen 0 after species form)
        if leaders:
            if len(leaders) > 1:
                leader_embeddings = np.array([emb for _, emb, _ in leaders])
                leader_phenotypes = [
                    pheno for _, _, pheno in leaders
                ]  # Keep as list to handle None values
                # ensemble_distances_batch handles None phenotypes by falling back to genotype-only
                distances = ensemble_distances_batch(
                    ind.embedding, leader_embeddings,
                    ind.phenotype, leader_phenotypes,
                    w_genotype, w_phenotype
                )
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                nearest_leader_id = leaders[min_idx][0]
            elif len(leaders) == 1:
                leader_emb = leaders[0][1]
                leader_pheno = leaders[0][2]
                min_dist = ensemble_distance(
                    ind.embedding, leader_emb,
                    ind.phenotype, leader_pheno,
                    w_genotype, w_phenotype
                )
                nearest_leader_id = leaders[0][0]
        
            # Assign to existing species if within threshold
            if nearest_leader_id is not None and min_dist < theta_sim:
                sp = species[nearest_leader_id]
                sp.add_member(ind)
                ind.species_id = nearest_leader_id
                species_with_new_members.add(nearest_leader_id)
                assigned = True
                # Update leader if this new member has higher fitness
                if ind.fitness > sp.leader.fitness:
                    if ind.fitness > sp.max_fitness:
                        sp.max_fitness = ind.fitness
                        sp.stagnation = 0
                    sp.leader = ind
                    sp.leader_distance = min_dist
                    for i, (sid, _, _) in enumerate(leaders):
                        if sid == nearest_leader_id:
                            leaders[i] = (sid, sp.leader.embedding, sp.leader.phenotype)
                            break
        
        # 2. For Generation 0: Check against potential leaders
        if not assigned and is_generation_0 and potential_leaders:
            for idx, (pl_species_id, pl_emb, pl_pheno, pl_ind) in enumerate(potential_leaders):
                dist = ensemble_distance(
                    ind.embedding, pl_emb,
                    ind.phenotype, pl_pheno,
                    w_genotype, w_phenotype
                )
                if dist < theta_sim:
                    if pl_species_id is None:
                        # First follower found! Create the species now
                        new_species_id = generate_species_id()
                        new_species = Species(
                            id=new_species_id,
                            leader=pl_ind,
                            members=[pl_ind, ind],
                            radius=theta_sim,
                            created_at=current_generation,
                            last_improvement=current_generation,
                            cluster_origin="natural",
                            parent_ids=None,
                            leader_distance=0.0
                        )
                        species[new_species_id] = new_species
                        pl_ind.species_id = new_species_id
                        ind.species_id = new_species_id
                        species_with_new_members.add(new_species_id)
                        # Update potential_leaders entry with new species ID
                        potential_leaders[idx] = (new_species_id, pl_emb, pl_pheno, pl_ind)
                        # Add to leaders list for future distance checks
                        leaders.append((new_species_id, pl_emb, pl_pheno))
                        logger.info(f"Species {new_species_id} formed: leader {pl_ind.id} + follower {ind.id}")
                    else:
                        # Species already exists (formed earlier), just add as member
                        sp = species[pl_species_id]
                        sp.add_member(ind)
                        ind.species_id = pl_species_id
                        species_with_new_members.add(pl_species_id)
                        # Update leader if higher fitness
                        if ind.fitness > sp.leader.fitness:
                            if ind.fitness > sp.max_fitness:
                                sp.max_fitness = ind.fitness
                                sp.stagnation = 0
                            sp.leader = ind
                            sp.leader_distance = dist
                    assigned = True
                    break
        
        # 3. For Generation N: Check against cluster 0 outliers (potential leaders)
        if not assigned and not is_generation_0 and cluster0_potential_leaders:
            for idx, (pl_species_id, pl_emb, pl_pheno, pl_ind) in enumerate(cluster0_potential_leaders):
                dist = ensemble_distance(
                    ind.embedding, pl_emb,
                    ind.phenotype, pl_pheno,
                    w_genotype, w_phenotype
                )
                if dist < theta_sim:
                    if pl_species_id is None:
                        # First follower found! Create the species now with both outlier and new individual
                        new_species_id = generate_species_id()
                        # Determine leader (highest fitness)
                        if ind.fitness > pl_ind.fitness:
                            leader = ind
                            follower = pl_ind
                        else:
                            leader = pl_ind
                            follower = ind
                        
                        new_species = Species(
                            id=new_species_id,
                            leader=leader,
                            members=[leader, follower],
                            radius=theta_sim,
                            created_at=current_generation,
                            last_improvement=current_generation,
                            cluster_origin="natural",
                            parent_ids=None,
                            leader_distance=0.0
                        )
                        species[new_species_id] = new_species
                        pl_ind.species_id = new_species_id
                        ind.species_id = new_species_id
                        species_with_new_members.add(new_species_id)
                        # Update cluster0_potential_leaders entry with new species ID
                        cluster0_potential_leaders[idx] = (new_species_id, pl_emb, pl_pheno, pl_ind)
                        # Add to leaders list for future distance checks
                        leaders.append((new_species_id, leader.embedding, leader.phenotype))
                        logger.info(f"Species {new_species_id} formed from cluster 0: leader {leader.id} + follower {follower.id}")
                    else:
                        # Species already exists (formed earlier), just add as member
                        sp = species[pl_species_id]
                        sp.add_member(ind)
                        ind.species_id = pl_species_id
                        species_with_new_members.add(pl_species_id)
                        # Update leader if higher fitness
                        if ind.fitness > sp.leader.fitness:
                            if ind.fitness > sp.max_fitness:
                                sp.max_fitness = ind.fitness
                                sp.stagnation = 0
                            sp.leader = ind
                            sp.leader_distance = dist
                            # Update leaders list
                            for i, (sid, _, _) in enumerate(leaders):
                                if sid == pl_species_id:
                                    leaders[i] = (sid, sp.leader.embedding, sp.leader.phenotype)
                                    break
                    assigned = True
                    break
        
        # 4. If not assigned to any species, potential leader, or cluster 0 outlier
        if not assigned:
            if is_generation_0:
                # Become a new potential leader
                ind.species_id = CLUSTER_0_ID
                potential_leaders.append((None, ind.embedding, ind.phenotype, ind))
                logger.debug(f"Individual {ind.id} becomes potential leader (cluster 0)")
            else:
                # Generation N: Not similar to species leaders or cluster 0 outliers -> add to cluster 0
                ind.species_id = CLUSTER_0_ID
                logger.debug(f"Individual {ind.id} outside all species and outliers -> added to cluster 0 (species_id=0)")
    
    # Log summary: potential leaders without followers stay in cluster 0
    if is_generation_0:
        unassigned_potential_leaders = sum(1 for pl_sid, _, _, _ in potential_leaders if pl_sid is None)
        if unassigned_potential_leaders > 0:
            logger.info(f"{unassigned_potential_leaders} potential leaders without followers -> stay in cluster 0")
    else:
        # Generation N: Log how many outliers formed species
        outliers_formed_species = sum(1 for pl_sid, _, _, _ in cluster0_potential_leaders if pl_sid is not None)
        outliers_still_in_cluster0 = sum(1 for pl_sid, _, _, _ in cluster0_potential_leaders if pl_sid is None)
        if outliers_formed_species > 0:
            logger.info(f"Generation N: {outliers_formed_species} cluster 0 outliers formed species")
        if outliers_still_in_cluster0 > 0:
            logger.debug(f"Generation N: {outliers_still_in_cluster0} cluster 0 outliers remain (no followers found)")
    
    # Update temp.json with species_id for each genome
    genome_id_to_species = {ind.id: ind.species_id for ind in valid_population}
    for genome in genomes:
        genome_id = genome.get("id")
        if genome_id in genome_id_to_species:
            genome["species_id"] = genome_id_to_species[genome_id]
        else:
            # Genome without embedding gets species_id=None
            genome["species_id"] = None
    
    # Save updated temp.json
    with open(temp_path_obj, 'w', encoding='utf-8') as f:
        json.dump(genomes, f, indent=2, ensure_ascii=False)
    
    # For Generation N: Update reserves.json for outliers that formed species
    if not is_generation_0 and cluster0_potential_leaders:
        outputs_path = get_outputs_path()
        reserves_path = outputs_path / "reserves.json"
        if reserves_path.exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                
                # Update species_id for outliers that formed species
                updated_count = 0
                for pl_species_id, _, _, pl_ind in cluster0_potential_leaders:
                    if pl_species_id is not None:  # This outlier formed a species
                        for genome in reserves_genomes:
                            if genome.get("id") == pl_ind.id:
                                genome["species_id"] = pl_species_id
                                updated_count += 1
                                logger.debug(f"Updated outlier {pl_ind.id} in reserves.json: species_id={pl_species_id}")
                                break
                
                if updated_count > 0:
                    # Save updated reserves.json
                    with open(reserves_path, 'w', encoding='utf-8') as f:
                        json.dump(reserves_genomes, f, indent=2, ensure_ascii=False)
                    logger.info(f"Updated {updated_count} outliers in reserves.json that formed species")
            except Exception as e:
                logger.warning(f"Failed to update reserves.json for outliers: {e}")
    
    # Update speciation_state.json with new species structure
    # Note: This only updates species, cluster0 is managed separately by SpeciationModule
    state_dict = {
        "species": {str(sid): sp.to_dict() for sid, sp in species.items()},
        "generation": current_generation
    }
    
    # If speciation_state.json exists, preserve cluster0 and other fields (but NOT config)
    if speciation_state_path_obj.exists():
        try:
            with open(speciation_state_path_obj, 'r', encoding='utf-8') as f:
                existing_state = json.load(f)
            # Preserve cluster0, global_best_id, metrics (config is not saved - it's passed as project args)
            state_dict["cluster0"] = existing_state.get("cluster0", {})
            state_dict["global_best_id"] = existing_state.get("global_best_id")
            state_dict["metrics"] = existing_state.get("metrics", {})
        except Exception:
            pass  # If can't read, just save new structure
    
    # Save updated speciation_state.json
    with open(speciation_state_path_obj, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Leader-Follower clustering: {len(valid_population)} individuals -> {len(species)} species")
    logger.debug(f"Species with new members: {sorted(species_with_new_members)}")
    return species, species_with_new_members
