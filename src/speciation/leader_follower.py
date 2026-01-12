"""
leader_follower.py

Leader-Follower clustering algorithm for speciation.
Reads temp.json and speciation_state.json directly, performs clustering,
and updates both files directly.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
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
    logger=None) -> Dict[int, Species]:
    """
    Leader-Follower clustering algorithm that reads and writes files directly.
    
    This function:
    1. Reads genomes from temp.json (must have prompt_embedding field)
    2. Reads existing species from speciation_state.json (if exists)
    3. Checks if any species exist (excluding species 0)
    4. If no species exist, treats as Generation 0:
       - Sorts genomes by fitness (descending)
       - Assigns highest fitness genome as leader of species 1
       - For each remaining genome, checks distance to all leaders
       - If within theta_sim, assigns to that species and updates leader if needed
       - If not, creates new species
       - After processing all, moves single-member species to species 0 (reserves)
    5. If species exist (Generation N):
       - Checks each genome against all existing species leaders
       - Also checks against reserves genomes (if speciation_state.json has cluster0)
       - Assigns to nearest species if within theta_sim, else creates new species
    6. Updates temp.json with species_id for each genome
    7. Updates speciation_state.json with new species structure
    
    Args:
        temp_path: Path to temp.json (defaults to outputs_path / "temp.json")
        speciation_state_path: Path to speciation_state.json (defaults to outputs_path / "speciation_state.json")
        theta_sim: Semantic distance threshold for species assignment
        current_generation: Current generation number
        logger: Optional logger instance
    
    Returns:
        Dict mapping species_id -> Species
    """
    if logger is None:
        logger = get_logger("LeaderFollowerClustering")
    
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
                    leader_phenotype = extract_phenotype_vector(sp_dict["leader_genome_data"])
                
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
                    cluster_origin=sp_dict.get("cluster_origin"),
                    parent_ids=sp_dict.get("parent_ids"),
                    parent_id=sp_dict.get("parent_id")
                )
                existing_species[sid] = species
            
            # Note: cluster0 members are not restored here since:
            # 1. Full genome data is stored in reserves.json, not speciation_state.json
            # 2. speciation_state.json only stores cluster0 metadata (size, speciation_events)
            # For Generation N, checking against reserves is optional and not implemented here.
            
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
    
    # Sort population by fitness (descending) - highest fitness processed first
    sorted_pop = sorted(valid_population, key=lambda x: x.fitness, reverse=True)
    
    # Initialize species dict
    species: Dict[int, Species] = existing_species.copy() if existing_species else {}
    leaders: List[Tuple[int, np.ndarray, Optional[np.ndarray]]] = [
        (sid, sp.leader.embedding, sp.leader.phenotype) 
        for sid, sp in species.items() 
        if sp.leader.embedding is not None
    ]
    
    # For Generation 0: First individual (highest fitness) becomes leader of species 1
    if is_generation_0:
        first = sorted_pop[0]
        first_species_id = 1  # Species IDs start from 1
        first_species = Species(
            id=first_species_id,
            leader=first,
            members=[first],
            radius=theta_sim,
            created_at=current_generation,
            last_improvement=current_generation,
            cluster_origin="natural",
            parent_ids=None,
            parent_id=None
        )
        species[first_species_id] = first_species
        leaders.append((first_species_id, first.embedding, first.phenotype))
        remaining_pop = sorted_pop[1:]
    else:
        # Generation N: process all individuals
        remaining_pop = sorted_pop
    
    # Process remaining individuals
    for ind in remaining_pop:
        min_dist = float('inf')
        nearest_leader_id = None
        
        # Find nearest leader from existing species using ensemble distance
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
        
        # Note: Checking against reserves genomes is not implemented here since:
        # 1. Reserves genomes are stored in reserves.json with full data
        # 2. Loading reserves.json for every genome check would be slow
        # 3. Reserves are outliers and creating new species for them happens via check_speciation()
        # For future: could load reserves.json once per generation if this check is needed
        
        # Decision: assign to species or create new species
        if nearest_leader_id is not None and min_dist < theta_sim:
            # Within threshold -> assign as follower
            sp = species[nearest_leader_id]
            sp.add_member(ind)
            ind.species_id = nearest_leader_id
            # Immediately update leader if this new member has higher fitness
            if ind.fitness > sp.leader.fitness:
                if ind.fitness > sp.max_fitness:
                    sp.max_fitness = ind.fitness
                    sp.stagnation = 0  # Reset stagnation when max_fitness improves
                sp.leader = ind
                # Update leader embedding in leaders list
                for i, (sid, _, _) in enumerate(leaders):
                    if sid == nearest_leader_id:
                        leaders[i] = (sid, sp.leader.embedding, sp.leader.phenotype)
                        break
        else:
            # Outside threshold -> create new species
            new_species_id = generate_species_id()
            new_species = Species(
                id=new_species_id,
                leader=ind,
                members=[ind],
                radius=theta_sim,
                created_at=current_generation,
                last_improvement=current_generation,
                cluster_origin="natural",
                parent_ids=None,
                parent_id=None
            )
            species[new_species_id] = new_species
            ind.species_id = new_species_id
            leaders.append((new_species_id, ind.embedding, ind.phenotype))
    
    # For Generation 0: Move single-member species to species 0 (reserves)
    # Minimum species size is 2 (leader + at least one follower)
    if is_generation_0:
        single_member_species = []
        for sid, sp in list(species.items()):
            if len(sp.members) == 1:
                # Only leader, no followers -> move to species 0
                single_member_species.append(sid)
                sp.members[0].species_id = CLUSTER_0_ID
        
        # Remove single-member species
        for sid in single_member_species:
            del species[sid]
            leaders = [(lid, emb, pheno) for lid, emb, pheno in leaders if lid != sid]
        
        if single_member_species:
            logger.info(f"Moved {len(single_member_species)} single-member species to reserves (minimum species size is 2)")
    
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
    return species
