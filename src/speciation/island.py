"""
island.py

Species/Island data structures for Dynamic Islands framework.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class IslandMode(Enum):
    """
    Operating mode for an island (species).
    
    Islands dynamically switch between modes based on fitness trends:
    - DEFAULT: Normal operation, balanced exploration/exploitation
    - EXPLORE: Increased diversity, relaxed selection, higher mutation
    - EXPLOIT: Focused exploitation, elite selection, lower mutation
    
    Mode switching is triggered by fitness slope analysis:
    - Positive slope → EXPLOIT (improving, focus on best)
    - Negative slope → EXPLORE (declining, need diversity)
    - Stable slope → DEFAULT (maintain current strategy)
    """
    DEFAULT = "DEFAULT"  # Balanced mode
    EXPLORE = "EXPLORE"   # Diversity-focused mode
    EXPLOIT = "EXPLOIT"   # Elite-focused mode


@dataclass
class Individual:
    """
    Represents an individual genome in the evolutionary population.
    
    An Individual is a wrapper around a genome (prompt) that includes:
    - Semantic embedding for clustering
    - Fitness score for selection
    - Species assignment for speciation
    
    This class bridges the gap between the raw genome format (dict)
    and the speciation framework's internal representation.
    
    Attributes:
        id: Unique identifier for the individual (matches genome ID)
        prompt: The text prompt (genome content)
        fitness: Fitness score (typically toxicity score, range [0, 1])
        embedding: L2-normalized semantic embedding vector (384-dim for all-MiniLM-L6-v2)
                   Used for semantic distance computation in clustering
        species_id: ID of the species this individual belongs to (None if unassigned, 0 for cluster 0)
        generation: Generation number when this individual was created
        genome_data: Original genome dictionary (for preserving metadata)
    """
    id: int
    prompt: str
    fitness: float = 0.0
    embedding: Optional[np.ndarray] = None
    species_id: Optional[int] = None
    generation: int = 0
    genome_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        Post-initialization: ensure embedding is numpy array.
        
        Converts embedding to numpy array if it's not already,
        which is required for distance computations.
        """
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)
    
    @classmethod
    def from_genome(cls, genome: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> "Individual":
        """
        Create Individual instance from genome dictionary.
        
        Extracts fitness from various possible locations in the genome dict:
        - "north_star_score" (primary)
        - "toxicity" (direct)
        - "scores"["toxicity"] (nested)
        - "moderation_result"["scores"]["toxicity"] (deeply nested)
        
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
        fitness = 0.0
        if "north_star_score" in genome:
            fitness = genome["north_star_score"]
        elif "toxicity" in genome:
            fitness = genome["toxicity"]
        elif "scores" in genome and isinstance(genome["scores"], dict):
            fitness = genome["scores"].get("toxicity", 0.0)
        elif "moderation_result" in genome and isinstance(genome["moderation_result"], dict):
            scores = genome["moderation_result"].get("scores", {})
            fitness = scores.get("toxicity", 0.0)
        
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
        
        return cls(
            id=genome.get("id", 0),
            prompt=genome.get("prompt", ""),
            fitness=float(fitness) if fitness else 0.0,
            embedding=final_embedding,
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
    Represents a species (island) in the Dynamic Islands framework.
    
    A Species is a cluster of semantically similar individuals that evolve together.
    Each species has:
    - A leader (highest fitness individual, defines species center)
    - Members (all individuals assigned to this species, max 100)
    - A radius (constant, equal to theta_sim for all species)
    - A mode (DEFAULT/EXPLORE/EXPLOIT) that controls evolutionary strategy
    - Fitness tracking for stagnation detection and mode switching
    - Origin tracking (how the species was created: merge/split/natural)
    
    Species evolve independently, can merge with similar species, can go extinct,
    and can split if they become too heterogeneous (low silhouette score).
    
    Note: Species IDs start from 1. ID 0 is reserved for Cluster 0 (reserves).
    
    Attributes:
        id: Unique species identifier (1+, 0 reserved for cluster 0)
        leader: Leader individual (highest fitness, defines species center)
        members: List of all individuals in this species (includes leader, max 100)
        mode: Current operating mode (DEFAULT/EXPLORE/EXPLOIT)
        radius: Semantic distance threshold for species membership (constant = theta_sim)
        stagnation_counter: Number of generations without improvement
        created_at: Generation when this species was created
        last_improvement: Generation when fitness last improved
        fitness_history: List of best fitness values over time (for trend analysis)
        cluster_origin: How this species was created ("merge", "split", "natural", or None)
        parent_ids: List of parent species IDs if created via merge [id1, id2]
        parent_id: ID of parent species if this was created via split (single parent)
        external_parent: External parent for EXPLORE mode (from cluster 0 or other species)
        mutation_rate: Current mutation rate multiplier (varies by mode)
        breeding_budget: Number of offspring to produce this generation
    """
    id: int
    leader: Individual
    members: List[Individual] = field(default_factory=list)
    mode: IslandMode = IslandMode.DEFAULT
    radius: float = 0.4  # Constant radius (theta_sim), no dynamic adjustment
    stagnation_counter: int = 0
    created_at: int = 0
    last_improvement: int = 0
    fitness_history: List[float] = field(default_factory=list)
    cluster_origin: Optional[str] = None  # "merge", "split", "natural", or None
    parent_ids: Optional[List[int]] = None  # For merged species: [parent1_id, parent2_id]
    parent_id: Optional[int] = None  # For split species: single parent ID
    external_parent: Optional[Individual] = None
    mutation_rate: float = 1.0
    breeding_budget: int = 5
    
    def __post_init__(self):
        """
        Post-initialization: ensure leader is in members and initialize tracking.
        
        - Ensures leader is always first in members list
        - Initializes fitness history with leader's fitness
        - Assigns species_id to all members
        """
        # Leader must be in members list (at position 0)
        if self.leader not in self.members:
            self.members.insert(0, self.leader)
        # Initialize fitness history if empty
        if not self.fitness_history and self.leader:
            self.fitness_history.append(self.leader.fitness)
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
    
    def update_leader(self) -> None:
        """
        Update leader to be the highest-fitness member.
        
        Leader defines the species center for clustering and distance computations.
        Should be called after fitness updates or member changes.
        """
        if self.members:
            self.leader = max(self.members, key=lambda x: x.fitness)
    
    def record_fitness(self, generation: int) -> None:
        """
        Record current best fitness and update stagnation tracking.
        
        Called each generation to track fitness trends for:
        - Stagnation detection (for extinction)
        - Mode switching (based on fitness slope)
        
        Args:
            generation: Current generation number
        """
        current_best = self.best_fitness
        # Check if we've improved
        if self.fitness_history and current_best > max(self.fitness_history):
            self.last_improvement = generation
            self.stagnation_counter = 0  # Reset stagnation counter
        else:
            self.stagnation_counter += 1  # Increment stagnation
        # Append to history for trend analysis
        self.fitness_history.append(current_best)
    
    def get_fitness_slope(self, window: int = 5) -> float:
        """
        Compute fitness trend slope over recent generations.
        
        Used for mode switching:
        - Positive slope → improving → switch to EXPLOIT
        - Negative slope → declining → switch to EXPLORE
        - Zero slope → stable → stay in DEFAULT
        
        Args:
            window: Number of recent generations to analyze
        
        Returns:
            Slope value (positive = improving, negative = declining)
        """
        if len(self.fitness_history) < window:
            return 0.0  # Not enough data
        recent = self.fitness_history[-window:]
        # Linear regression slope: (y_end - y_start) / (x_end - x_start)
        return (recent[-1] - recent[0]) / (window - 1) if window > 1 else 0.0
    
    def get_sorted_members(self, reverse: bool = True) -> List[Individual]:
        """
        Get members sorted by fitness.
        
        Args:
            reverse: If True, sort descending (highest first), else ascending
        
        Returns:
            Sorted list of members
        """
        return sorted(self.members, key=lambda x: x.fitness, reverse=reverse)
    
    def enforce_capacity(self, max_capacity: int = 100) -> List[Individual]:
        """
        Enforce maximum capacity by removing lowest-fitness members.
        
        Keeps the top `max_capacity` individuals by fitness.
        Removed individuals should be sent to cluster 0.
        
        Args:
            max_capacity: Maximum number of members to keep (default: 100)
        
        Returns:
            List of removed individuals (to be sent to cluster 0)
        """
        if self.size <= max_capacity:
            return []
        
        # Sort by fitness (highest first) and keep top max_capacity
        sorted_members = self.get_sorted_members(reverse=True)
        keep = sorted_members[:max_capacity]
        removed = sorted_members[max_capacity:]
        
        # Update members list
        self.members = keep
        
        # Clear species_id for removed individuals
        for ind in removed:
            ind.species_id = None
        
        # Ensure leader is still in the list
        if self.leader not in self.members and self.members:
            self.leader = self.members[0]
        
        return removed
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize species to dictionary for JSON storage.
        
        Includes leader embedding for state restoration across generations.
        Includes cluster_origin and parent_ids for origin tracking.
        """
        return {
            "id": self.id,
            "leader_id": self.leader.id,
            "leader_prompt": self.leader.prompt,
            "leader_embedding": self.leader.embedding.tolist() if self.leader.embedding is not None else None,
            "leader_fitness": self.leader.fitness,
            "member_ids": [m.id for m in self.members],
            "mode": self.mode.value,
            "radius": self.radius,
            "stagnation_counter": self.stagnation_counter,
            "created_at": self.created_at,
            "last_improvement": self.last_improvement,
            "fitness_history": self.fitness_history[-20:],
            "cluster_origin": self.cluster_origin,
            "parent_ids": self.parent_ids,
            "parent_id": self.parent_id,
            "size": self.size,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
        }
    
    def __repr__(self):
        origin_str = f", origin={self.cluster_origin}" if self.cluster_origin else ""
        return f"Species(id={self.id}, size={self.size}, mode={self.mode.value}, best={self.best_fitness:.4f}{origin_str})"


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
