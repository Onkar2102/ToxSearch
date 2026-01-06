"""
island.py

Species/Island data structures for Plan A+ Dynamic Islands framework.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class IslandMode(Enum):
    """Operating mode for an island."""
    DEFAULT = "DEFAULT"
    EXPLORE = "EXPLORE"
    EXPLOIT = "EXPLOIT"


@dataclass
class Individual:
    """Represents an individual with embedding and fitness."""
    id: int
    prompt: str
    fitness: float = 0.0
    embedding: Optional[np.ndarray] = None
    species_id: Optional[int] = None
    generation: int = 0
    genome_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)
    
    @classmethod
    def from_genome(cls, genome: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> "Individual":
        """Create Individual from genome dictionary."""
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
        
        return cls(
            id=genome.get("id", 0),
            prompt=genome.get("prompt", ""),
            fitness=float(fitness) if fitness else 0.0,
            embedding=embedding,
            species_id=genome.get("species_id"),
            generation=genome.get("generation", 0),
            genome_data=genome
        )
    
    def to_genome(self) -> Dict[str, Any]:
        """Convert back to genome dictionary."""
        genome = self.genome_data.copy() if self.genome_data else {"id": self.id, "prompt": self.prompt}
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
    """Represents a species (island) in the Plan A+ framework."""
    id: int
    leader: Individual
    members: List[Individual] = field(default_factory=list)
    mode: IslandMode = IslandMode.DEFAULT
    radius: float = 0.4
    stagnation_counter: int = 0
    created_at: int = 0
    last_improvement: int = 0
    fitness_history: List[float] = field(default_factory=list)
    parent_id: Optional[int] = None
    external_parent: Optional[Individual] = None
    mutation_rate: float = 1.0
    breeding_budget: int = 5
    
    def __post_init__(self):
        if self.leader not in self.members:
            self.members.insert(0, self.leader)
        if not self.fitness_history and self.leader:
            self.fitness_history.append(self.leader.fitness)
        for member in self.members:
            member.species_id = self.id
    
    @property
    def size(self) -> int:
        return len(self.members)
    
    @property
    def best_fitness(self) -> float:
        return max((m.fitness for m in self.members), default=0.0)
    
    @property
    def avg_fitness(self) -> float:
        return sum(m.fitness for m in self.members) / len(self.members) if self.members else 0.0
    
    @property
    def leader_embedding(self) -> Optional[np.ndarray]:
        return self.leader.embedding if self.leader else None
    
    def add_member(self, individual: Individual) -> None:
        individual.species_id = self.id
        if individual not in self.members:
            self.members.append(individual)
    
    def remove_member(self, individual: Individual) -> bool:
        if individual in self.members:
            self.members.remove(individual)
            individual.species_id = None
            return True
        return False
    
    def update_leader(self) -> None:
        if self.members:
            self.leader = max(self.members, key=lambda x: x.fitness)
    
    def record_fitness(self, generation: int) -> None:
        current_best = self.best_fitness
        if self.fitness_history and current_best > max(self.fitness_history):
            self.last_improvement = generation
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.fitness_history.append(current_best)
    
    def get_fitness_slope(self, window: int = 5) -> float:
        if len(self.fitness_history) < window:
            return 0.0
        recent = self.fitness_history[-window:]
        return (recent[-1] - recent[0]) / (window - 1) if window > 1 else 0.0
    
    def get_sorted_members(self, reverse: bool = True) -> List[Individual]:
        return sorted(self.members, key=lambda x: x.fitness, reverse=reverse)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "leader_id": self.leader.id,
            "member_ids": [m.id for m in self.members],
            "mode": self.mode.value,
            "radius": self.radius,
            "stagnation_counter": self.stagnation_counter,
            "created_at": self.created_at,
            "last_improvement": self.last_improvement,
            "fitness_history": self.fitness_history[-20:],
            "size": self.size,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
        }
    
    def __repr__(self):
        return f"Species(id={self.id}, size={self.size}, mode={self.mode.value}, best={self.best_fitness:.4f})"


class SpeciesIdGenerator:
    """Thread-safe species ID generator."""
    _current_id: int = 0
    
    @classmethod
    def next_id(cls) -> int:
        cls._current_id += 1
        return cls._current_id
    
    @classmethod
    def reset(cls, start: int = 0) -> None:
        cls._current_id = start


def generate_species_id() -> int:
    return SpeciesIdGenerator.next_id()

