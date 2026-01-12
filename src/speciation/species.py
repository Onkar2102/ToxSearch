"""
species.py

Species data structures for speciation framework.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


# Note: SpeciesMode/IslandMode removed - mode switching is handled by parent selection, not speciation


@dataclass
class Individual:
    """
    Represents an individual genome in the evolutionary population.
    
    An Individual is a wrapper around a genome (prompt) that includes:
    - Genotype: Semantic embedding for clustering (prompt embedding)
    - Phenotype: Response scores (8D toxicity scores)
    - Fitness score for selection
    - Species assignment for speciation
    
    This class bridges the gap between the raw genome format (dict)
    and the speciation framework's internal representation.
    
    Attributes:
        id: Unique identifier for the individual (matches genome ID)
        prompt: The text prompt (genome content)
        fitness: Fitness score (typically toxicity score, range [0, 1])
        embedding: L2-normalized semantic embedding vector (384-dim for all-MiniLM-L6-v2)
                   Used for genotype distance computation in clustering
        phenotype: Phenotype vector (8D response scores) for phenotype distance computation
        species_id: ID of the species this individual belongs to (None if unassigned, 0 for cluster 0)
        generation: Generation number when this individual was created
        genome_data: Original genome dictionary (for preserving metadata)
    """
    id: int
    prompt: str
    fitness: float = 0.0
    embedding: Optional[np.ndarray] = None
    phenotype: Optional[np.ndarray] = None
    species_id: Optional[int] = None
    generation: int = 0
    genome_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        Post-initialization: ensure embedding and phenotype are numpy arrays.
        
        Converts embedding and phenotype to numpy arrays if they're not already,
        which is required for distance computations.
        """
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)
        
        if self.phenotype is not None and not isinstance(self.phenotype, np.ndarray):
            self.phenotype = np.array(self.phenotype)
    
    @classmethod
    def from_genome(cls, genome: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> "Individual":
        """
        Create Individual instance from genome dictionary.
        
        Extracts fitness from various possible locations in the genome dict (priority order):
        - "north_star_score" (primary - pre-computed metric)
        - "moderation_result"["google"]["scores"]["toxicity"] (standard format)
        - "moderation_result"["scores"]["toxicity"] (legacy format)
        - "toxicity" (direct)
        - "scores"["toxicity"] (nested)
        
        Extracts embedding from "prompt_embedding" field if present (preferred),
        otherwise uses provided embedding parameter.
        
        Args:
            genome: Genome dictionary with prompt, id, fitness, and optionally prompt_embedding
            embedding: Optional pre-computed embedding (used only if prompt_embedding not in genome)
        
        Returns:
            Individual instance with extracted data
        
        Example:
            >>> genome = {"id": 1, "prompt": "test", "toxicity": 0.8, "prompt_embedding": [0.1, 0.2, ...]}
            >>> ind = Individual.from_genome(genome)
        """
        # Try multiple locations for fitness score (handles different genome formats)
        # Priority: north_star_score > moderation_result.google.scores > toxicity > scores.toxicity
        fitness = 0.0
        if "north_star_score" in genome:
            fitness = genome["north_star_score"]
        elif "moderation_result" in genome and isinstance(genome["moderation_result"], dict):
            # Check moderation_result.google.scores.toxicity (standard format)
            google_result = genome["moderation_result"].get("google", {})
            if google_result and "scores" in google_result:
                fitness = google_result["scores"].get("toxicity", 0.0)
            else:
                # Fallback to moderation_result.scores.toxicity (legacy format)
                scores = genome["moderation_result"].get("scores", {})
                fitness = scores.get("toxicity", 0.0)
        elif "toxicity" in genome:
            fitness = genome["toxicity"]
        elif "scores" in genome and isinstance(genome["scores"], dict):
            fitness = genome["scores"].get("toxicity", 0.0)
        
        # Extract embedding from genome if present (preferred over parameter)
        final_embedding = embedding
        if "prompt_embedding" in genome:
            # Convert list to numpy array
            embedding_list = genome["prompt_embedding"]
            if isinstance(embedding_list, list):
                final_embedding = np.array(embedding_list)
            elif isinstance(embedding_list, np.ndarray):
                final_embedding = embedding_list
        elif embedding is not None:
            final_embedding = embedding
        
        # Extract phenotype vector (response scores)
        from .phenotype_distance import extract_phenotype_vector
        phenotype = extract_phenotype_vector(genome)
        
        return cls(
            id=genome.get("id", 0),
            prompt=genome.get("prompt", ""),
            fitness=float(fitness) if fitness else 0.0,
            embedding=final_embedding,
            phenotype=phenotype,
            species_id=genome.get("species_id"),
            generation=genome.get("generation", 0),
            genome_data=genome
        )
    
    def to_genome(self) -> Dict[str, Any]:
        """
        Convert Individual back to genome dictionary format.
        
        Updates the original genome dict with speciation information:
        - species_id: Which species this individual belongs to (0 = cluster 0)
        - fitness: Current fitness value
        
        Returns:
            Genome dictionary with updated speciation metadata.
            This is used to update genomes after speciation processing.
        """
        # Start with original genome data or create minimal dict
        genome = self.genome_data.copy() if self.genome_data else {"id": self.id, "prompt": self.prompt}
        # Add speciation metadata
        genome["species_id"] = self.species_id
        genome["fitness"] = self.fitness
        return genome
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Individual) and self.id == other.id
    
    def __repr__(self):
        return f"Individual(id={self.id}, fitness={self.fitness:.4f}, species_id={self.species_id})"


@dataclass
class Species:
    """
    Represents a species in the speciation framework.
    
    A Species is a cluster of semantically similar individuals that evolve together.
    Each species has:
    - A leader (highest fitness individual, defines species center)
    - Members (all individuals assigned to this species, max species_capacity)
    - A radius (constant, equal to theta_sim for all species)
    - A mode field (kept for backward compatibility, always DEFAULT)
    - Fitness tracking for stagnation detection
    - Origin tracking (how the species was created: merge/split/natural)
    
    Species evolve independently, can merge with similar species, can go extinct.
    
    Note: Species IDs start from 1. ID 0 is reserved for Cluster 0 (reserves).
    
    Attributes:
        id: Unique species identifier (1+, 0 reserved for cluster 0)
        leader: Leader individual (highest fitness, defines species center)
        members: List of all individuals in this species (includes leader, max species_capacity)
        mode: Operating mode (always DEFAULT, kept for backward compatibility)
        radius: Semantic distance threshold for species membership (constant = theta_sim)
        stagnation: Number of generations without max_fitness improvement
        max_fitness: Current maximum fitness score in this species
        species_state: "active", "stagnant", or "frozen" (frozen/stagnant species not used for parent selection)
        created_at: Generation when this species was created
        last_improvement: Generation when fitness last improved
        fitness_history: List of best fitness values over time (for trend analysis)
        labels: Current c-TF-IDF labels (top 10 representative words)
        label_history: History of labels over generations (for tracking topic evolution)
        cluster_origin: How this species was created ("merge", "split", "natural", or None)
        parent_ids: List of parent species IDs if created via merge [id1, id2]
        parent_id: ID of parent species if this was created via split (single parent)
    """
    id: int
    leader: Individual
    members: List[Individual] = field(default_factory=list)
    radius: float = 0.4  # Constant radius (theta_sim), no dynamic adjustment
    stagnation: int = 0  # Generations without max_fitness improvement
    max_fitness: float = 0.0  # Current maximum fitness in this species
    species_state: str = "active"  # "active", "stagnant", or "frozen"
    created_at: int = 0
    last_improvement: int = 0
    fitness_history: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)  # Current c-TF-IDF labels (10 words)
    label_history: List[Dict[str, Any]] = field(default_factory=list)  # Label history per generation
    cluster_origin: Optional[str] = None  # "merge", "split", "natural", or None
    parent_ids: Optional[List[int]] = None  # For merged species: [parent1_id, parent2_id]
    parent_id: Optional[int] = None  # For split species: single parent ID
    
    def __post_init__(self):
        """
        Post-initialization: ensure leader is in members and initialize tracking.
        
        - Ensures leader is always first in members list
        - Initializes fitness history with leader's fitness
        - Initializes max_fitness with leader's fitness
        - Assigns species_id to all members
        """
        # Leader must be in members list (at position 0)
        if self.leader not in self.members:
            self.members.insert(0, self.leader)
        # Initialize fitness history and max_fitness if empty
        if not self.fitness_history and self.leader:
            self.fitness_history.append(self.leader.fitness)
        if self.max_fitness == 0.0 and self.leader:
            self.max_fitness = self.leader.fitness
        # Assign species_id to all members
        for member in self.members:
            member.species_id = self.id
    
    @property
    def size(self) -> int:
        """Number of individuals in this species."""
        return len(self.members)
    
    @property
    def best_fitness(self) -> float:
        """Highest fitness value in this species."""
        return max((m.fitness for m in self.members), default=0.0)
    
    @property
    def avg_fitness(self) -> float:
        """Average fitness across all members."""
        return sum(m.fitness for m in self.members) / len(self.members) if self.members else 0.0
    
    @property
    def leader_embedding(self) -> Optional[np.ndarray]:
        """Leader's embedding vector (for distance computations)."""
        return self.leader.embedding if self.leader else None
    
    def add_member(self, individual: Individual) -> None:
        """
        Add an individual to this species.
        
        Args:
            individual: Individual to add (species_id is automatically set)
        """
        individual.species_id = self.id
        if individual not in self.members:
            self.members.append(individual)
    
    def remove_member(self, individual: Individual) -> bool:
        """
        Remove an individual from this species.
        
        Args:
            individual: Individual to remove
        
        Returns:
            True if removed, False if not found
        """
        if individual in self.members:
            self.members.remove(individual)
            individual.species_id = None
            return True
        return False
    
    def record_fitness(self, generation: int) -> None:
        """
        Record current best fitness and update fitness history.
        
        Args:
            generation: Current generation number
        """
        current_best = self.best_fitness
        # Check if we've improved
        if self.fitness_history and current_best > max(self.fitness_history):
            self.last_improvement = generation
        # Append to history for trend analysis
        self.fitness_history.append(current_best)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize species to dictionary for JSON storage.
        
        Includes leader embedding for state restoration across generations.
        Includes cluster_origin and parent_ids for origin tracking.
        Includes labels and label_history for semantic characterization.
        
        Note: Computed fields (size, best_fitness, avg_fitness) are not included
        as they can be derived from members and are not used by load_state().
        """
        return {
            "id": self.id,
            "leader_id": self.leader.id,
            "leader_prompt": self.leader.prompt,
            "leader_embedding": self.leader.embedding.tolist() if self.leader.embedding is not None else None,
            "leader_fitness": self.leader.fitness,
            "member_ids": [m.id for m in self.members],
            "radius": self.radius,
            "stagnation": self.stagnation,
            "max_fitness": self.max_fitness,
            "species_state": self.species_state,
            "created_at": self.created_at,
            "last_improvement": self.last_improvement,
            "fitness_history": self.fitness_history[-20:],
            "labels": self.labels,
            "label_history": self.label_history[-20:],  # Keep last 20 generations
            "cluster_origin": self.cluster_origin,
            "parent_ids": self.parent_ids,
            "parent_id": self.parent_id,
        }
    
    def __repr__(self):
        origin_str = f", origin={self.cluster_origin}" if self.cluster_origin else ""
        return f"Species(id={self.id}, size={self.size}, best={self.best_fitness:.4f}{origin_str})"


class SpeciesIdGenerator:
    """
    Thread-safe species ID generator (singleton pattern).
    
    Ensures unique species IDs across the entire evolution run.
    IDs are sequential integers starting from 1 (ID 0 is reserved for cluster 0).
    """
    _current_id: int = 0  # Class variable (shared across all instances)
    
    @classmethod
    def next_id(cls) -> int:
        """
        Generate next unique species ID.
        
        Returns:
            Next sequential species ID (starts from 1, 0 reserved for cluster 0)
        """
        cls._current_id += 1
        return cls._current_id
    
    @classmethod
    def reset(cls, start: int = 0) -> None:
        """
        Reset ID counter (useful for testing or fresh runs).
        
        Args:
            start: Starting ID value (default: 0, so first ID will be 1)
        """
        cls._current_id = start
    
    @classmethod
    def set_min_id(cls, min_id: int) -> None:
        """
        Ensure ID counter is at least min_id (for state restoration).
        
        Used when loading species from saved state to avoid ID conflicts.
        
        Args:
            min_id: Minimum ID value to set (counter will be max of current and min_id)
        """
        cls._current_id = max(cls._current_id, min_id)


def generate_species_id() -> int:
    """
    Convenience function to generate a new species ID.
    
    Returns:
        New unique species ID (1+, 0 is reserved for cluster 0)
    """
    return SpeciesIdGenerator.next_id()
