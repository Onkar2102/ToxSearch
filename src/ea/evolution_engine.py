"""
EvolutionEngine.py

Core evolutionary algorithm engine for text generation optimization.

This module implements the main evolution logic including parent selection,
operator application, and population management. Uses steady-state evolution
with elites preservation and multi-API moderation scoring.
"""

import json
import random
from typing import List, Dict, Any, Optional
from utils import get_custom_logging
from utils.population_io import _extract_north_star_score
from .parent_selector import ParentSelector
from itertools import combinations
from pathlib import Path
from .synonym_replacement import LLM_POSAwareSynonymReplacement
from .mlm_operator import MLMOperator
from .paraphrasing import LLMBasedParaphrasingOperator
from .antonym_replacement import POSAwareAntonymReplacement
from .stylistic_mutator import StylisticMutator
from .back_translation import (
    LLMBackTranslationHIOperator, LLMBackTranslationFROperator, 
    LLMBackTranslationDEOperator, LLMBackTranslationJAOperator, 
    LLMBackTranslationZHOperator
)
from .semantic_similarity_crossover import SemanticSimilarityCrossover
from .fusion_crossover import SemanticFusionCrossover
from .negation_operator import NegationOperator
from .typographical_errors import TypographicalErrorsOperator
from .concept_addition import ConceptAdditionOperator
from .informed_evolution import InformedEvolutionOperator

# Global generator instances - will be set by main.py
_global_response_generator = None
_global_prompt_generator = None

def set_global_generators(response_generator, prompt_generator):
    """Set the global generator instances to be used by the system."""
    global _global_response_generator, _global_prompt_generator
    _global_response_generator = response_generator
    _global_prompt_generator = prompt_generator

def get_response_generator():
    """Get the shared response generator instance."""
    global _global_response_generator
    if _global_response_generator is None:
        raise RuntimeError("No global response generator set. Call set_global_generators() first from main.py")
    return _global_response_generator

def get_prompt_generator():
    """Get the shared prompt generator instance."""
    global _global_prompt_generator
    if _global_prompt_generator is None:
        raise RuntimeError("No global prompt generator set. Call set_global_generators() first from main.py")
    return _global_prompt_generator

# Legacy function for backward compatibility
def get_generator():
    """Get the shared prompt generator instance (legacy function)."""
    return get_prompt_generator()

class EvolutionEngine:

    def __init__(self, north_star_metric, log_file, current_cycle=None, max_variants=3, adaptive_selection_after=5, max_num_parents=4, operators="all", outputs_path=None):
        self._genomes_loaded = False
        self._genomes_cache = []
        self.next_id = 0
        self.north_star_metric = north_star_metric
        self.log_file = log_file
        self.current_cycle = current_cycle  # Current evolution cycle number
        self.use_steady_state = True
        self.max_variants = max_variants  # Maximum number of variants to generate per operator
        self.operators = operators  # Operator configuration mode: "ie", "cm", or "all"
        self.outputs_path = outputs_path  # Output directory path for this run
        get_logger, _, _, _ = get_custom_logging()
        self.logger = get_logger("EvolutionEngine", log_file)
        self.parent_selector = ParentSelector(north_star_metric, log_file, adaptive_selection_after=adaptive_selection_after, max_num_parents=max_num_parents)
        # Initialize the shared generator instances
        self.prompt_generator = get_prompt_generator()
        self.response_generator = get_response_generator()
        
        self.logger.debug(f"EvolutionEngine initialized with next_id={self.next_id}, north_star_metric={north_star_metric}, current_cycle={current_cycle}, max_variants={max_variants}, adaptive_selection_after={adaptive_selection_after}, max_num_parents={max_num_parents}, operators={operators}, use_steady_state=True")

    @property
    def genomes(self):
        """Lazy load genomes only when needed"""
        if not self._genomes_loaded:
            # Load from files only when first accessed
            from utils.population_io import load_population
            self._genomes_cache = load_population(str(self.outputs_path), logger=self.logger)
            self._genomes_loaded = True
            self.logger.debug(f"Lazy loaded {len(self._genomes_cache)} genomes")
        return self._genomes_cache

    @genomes.setter
    def genomes(self, value):
        """Allow setting genomes directly"""
        self._genomes_cache = value
        self._genomes_loaded = True

    def update_next_id(self):
        """Update next_id based on current genomes."""
        if self.genomes:
            max_id = max(g["id"] for g in self.genomes if g.get("id") is not None)
            self.next_id = max_id + 1
        else:
            self.next_id = 1
        self.logger.debug(f"Updated next_id to {self.next_id}")
    

    def _count_variants_from_temp(self) -> Dict[str, int]:
        """Count variants in temp.json by type (mutation/crossover)."""
        try:
            temp_path = Path(self.outputs_path) / "temp.json"
            if not temp_path.exists():
                return {"mutation_variants": 0, "crossover_variants": 0, "variants_created": 0}
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                temp_variants = json.load(f)
            
            mutation_count = sum(1 for v in temp_variants if v and v.get("creation_info", {}).get("type") == "mutation")
            crossover_count = sum(1 for v in temp_variants if v and v.get("creation_info", {}).get("type") == "crossover")
            total_count = mutation_count + crossover_count
            
            self.logger.debug(f"Counted variants from temp.json: {mutation_count} mutation, {crossover_count} crossover, {total_count} total")
            
            return {
                "mutation_variants": mutation_count,
                "crossover_variants": crossover_count,
                "variants_created": total_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to count variants from temp.json: {e}")
            return {"mutation_variants": 0, "crossover_variants": 0, "variants_created": 0}

    def _analyze_generation_data(self, parents: List[Dict], variant_counts: Dict[str, int]) -> Dict[str, Any]:
        """Analyze and create generation data for tracking and analytics."""
        generation_data = {
            "generation_number": self.current_cycle,
            "parents": [],
            "variants_created": variant_counts["variants_created"],
            "mutation_variants": variant_counts["mutation_variants"],
            "crossover_variants": variant_counts["crossover_variants"]
        }
        
        # Record selected parent metadata
        for parent in parents:
            generation_data["parents"].append({
                "id": parent["id"],
                "north_star_score": round(_extract_north_star_score(parent, "toxicity"), 4),
                "generation": parent["generation"],
                "type": "parent"
            })
        
        return generation_data

    def _calculate_parent_score(self, parents: List[Dict], variant_type: str, operator: Any = None) -> float:
        """
        Calculate parent score based on variant type.
        
        Args:
            parents: List of parent genomes (simplified structure with 'toxicity' field)
            variant_type: Type of variant ("mutation" or "crossover")
            operator: Operator instance (for InformedEvolutionOperator special handling)
            
        Returns:
            float: Parent score (minimum 0.0001 for consistency)
        """
        # Special handling for InformedEvolutionOperator - use top_10 average
        if operator and hasattr(operator, 'top_10_avg_score'):
            self.logger.debug(f"Using top_10 average score: {operator.top_10_avg_score:.4f}")
            return operator.top_10_avg_score
        
        # Parents from parents.json have simplified structure with direct 'toxicity' field
        if variant_type == "mutation":
            # For mutation, use the single parent's score (minimum 0.0001)
            if not parents:
                return 0.0001
            parent_score = parents[0].get("toxicity", 0.0001)
            # Ensure minimum score
            return max(round(parent_score, 4), 0.0001)
        elif variant_type == "crossover":
            # For crossover, average ALL parents' scores (don't filter out any parent)
            if not parents:
                return 0.0001
            # Include all parents, use default 0.0001 if toxicity missing
            scores = [max(p.get("toxicity", 0.0001), 0.0001) for p in parents]
            avg_score = sum(scores) / len(scores)
            return round(avg_score, 4)
        
        return 0.0001

    def _create_child_genome(self, prompt: str, operator: Any, parents: List[Dict], variant_type: str) -> Dict:
        """Create a child genome from a prompt and operator."""
        # Calculate parent score (average) once for both top-level and creation_info
        parent_score = self._calculate_parent_score(parents, variant_type, operator)
        
        # Store parent information with both ID and score
        parents_info = []
        for p in parents:
            parent_id = p.get("id")
            # Get the toxicity score for this parent
            parent_toxicity = p.get("toxicity", 0.0001)
            parents_info.append({
                "id": parent_id,
                "score": round(parent_toxicity, 4)
            })
        
        child = {
            "id": self.next_id,
            "prompt": prompt,
            "model_name": None,
            "moderation_result": None,
            "operator": operator.name,
            "parents": parents_info,  # List of {id, score} objects
            "generation": self.current_cycle,
            "status": "pending_generation",
            "parent_score": parent_score,  # Average/calculated parent score
            "creation_info": {
                "type": variant_type,
                "operator": operator.name,
                "source_generation": max(p.get("generation", 0) for p in parents) if len(parents) > 1 else parents[0].get("generation", 0),
                "evolution_cycle": self.current_cycle,
                "parent_score": parent_score  # Also kept in creation_info for backward compatibility
            }
        }
        
        # Add operator timing if available
        if hasattr(operator, '_last_operation_time'):
            child['variant_creation_duration'] = operator._last_operation_time.get('duration', 0.0)
        
        self.next_id += 1
        return child

    def generate_variants_global(self, evolution_tracker: Dict[str, Any] = None) -> None:
        """
        Generate variants globally for evolution cycle.
        Updates temp.json with unique variants created.
        
        Args:
            evolution_tracker (Dict[str, Any]): Evolution tracker data for determining parent counts
        """
        self.logger.debug(f"Generating variants globally for evolution cycle {self.current_cycle}")

        # Step 1: Synchronize next_id with current genomes
        # Prevents ID reuse if engine persists across cycles.
        self.update_next_id()

        # Step 2: Handle different file usage patterns based on operator mode
        if self.operators == "ie":
            # Mode "ie": Only use top_10.json, skip parent selection
            self.logger.info("Operator mode 'ie': Skipping parent selection, using only top_10.json")
            self._generate_variants_ie_mode(evolution_tracker)
        elif self.operators == "cm":
            # Mode "cm": Only use parents.json, skip top_10.json
            self.logger.info("Operator mode 'cm': Using parents.json, skipping top_10.json")
            self._generate_variants_cm_mode(evolution_tracker)
        elif self.operators == "all":
            # Mode "all": Use both files (default behavior)
            self.logger.info("Operator mode 'all': Using both parents.json and top_10.json")
            self._generate_variants_all_mode(evolution_tracker)
        else:
            # Default to all mode
            self.logger.warning(f"Unknown operator mode '{self.operators}', defaulting to 'all'")
            self._generate_variants_all_mode(evolution_tracker)
        
        # Step 3: EvolutionTracker updates are handled by parent_selector.py only
        # No additional updates needed here to avoid conflicts

    def _generate_variants_ie_mode(self, evolution_tracker: Dict[str, Any] = None) -> None:
        """Generate variants using only InformedEvolution operator with top_10.json"""
        self.logger.info("Running IE mode: Using only InformedEvolution operator with top_10.json")
        
        # First, populate top_10.json with the most toxic examples from elites and population
        try:
            elites_path = str(Path(self.outputs_path) / "elites.json")
            top_10_path = str(Path(self.outputs_path) / "top_10.json")
            self.parent_selector._save_top_10_by_toxicity(elites_path, top_10_path)
            self.logger.info("Populated top_10.json with most toxic examples for IE mode")
            
            # top_10.json is already populated by parent_selector.py
            # EvolutionTracker is already updated by parent_selector.py
                
        except Exception as e:
            self.logger.error(f"Failed to populate top_10.json: {e}")
            return
        
        # Get only InformedEvolution operator
        ie_operators = self._get_single_parent_operators()
        
        if not ie_operators:
            self.logger.error("No InformedEvolution operators found in IE mode")
            return
        
        # Load a real parent from top_10.json for proper genome creation
        top_10_path = Path(self.outputs_path) / "top_10.json"
        parent_example = None
        
        if top_10_path.exists():
            with open(top_10_path, 'r', encoding='utf-8') as f:
                top_10_examples = json.load(f)
            if top_10_examples:
                # Use the first example as parent (InformedEvolution reads from top_10.json internally anyway)
                parent_example = top_10_examples[0]
                self.logger.debug(f"Using parent example from top_10.json: {parent_example['id']}")
            else:
                self.logger.error("No examples found in top_10.json")
                return
        else:
            self.logger.error("top_10.json not found")
            return
        
        # Run InformedEvolution operator max_variants times
        for operator in ie_operators:
            try:
                self.logger.debug(f"Running operator: {operator.__class__.__name__} {self.max_variants} times")
                
                variants_to_save = []
                for variant_iteration in range(self.max_variants):
                    operator_input = {
                        "parent_data": parent_example
                    }
                    
                    # Apply operator
                    variants = operator.apply(operator_input)
                    
                    if variants:
                        # Create child genomes
                        variants_to_save.extend([self._create_child_genome(vp, operator, [parent_example], "mutation") for vp in variants])
                
                if variants_to_save:
                    # Save all variants to temp.json
                    self._append_variants_to_temp(variants_to_save)
                    self.logger.info(f"Generated {len(variants_to_save)} variants using {operator.__class__.__name__} ({self.max_variants} calls)")
                    self.logger.debug(f"Saved {len(variants_to_save)} variants to temp.json")
                else:
                    self.logger.warning(f"No variants generated by {operator.__class__.__name__}")
                    
            except Exception as e:
                self.logger.error(f"Error running operator {operator.__class__.__name__}: {e}", exc_info=True)
        
        # Update EvolutionTracker and clean files after all operators have processed
        parents_path = Path(self.outputs_path) / "parents.json"
        self._update_evolution_tracker_from_files(parents_path, top_10_path)
        # Note: In IE mode, we only clean top_10.json (parents.json may not exist)
        try:
            if top_10_path.exists():
                with open(top_10_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
                self.logger.info(f"Emptied file: {top_10_path}")
        except Exception as e:
            self.logger.error(f"Failed to empty top_10 file: {e}")

    def _generate_variants_cm_mode(self, evolution_tracker: Dict[str, Any] = None) -> None:
        """Generate variants using all operators except InformedEvolution, using parents.json"""
        self.logger.info("Running CM mode: Using all operators except InformedEvolution with parents.json")
        
        # Validate that elites exist before attempting parent selection
        elites_path = Path(self.outputs_path) / "elites.json"
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
            if not elites:
                self.logger.error("CRITICAL ERROR: elites.json exists but is empty - this indicates a fundamental problem")
                self.logger.error("Evolution cannot continue without elites. Stopping immediately.")
                raise RuntimeError("Empty elites.json - evolution cannot continue. This indicates a critical system failure.")
        else:
            self.logger.error("CRITICAL ERROR: elites.json does not exist - this indicates a fundamental problem")
            self.logger.error("Evolution cannot continue without elites. Stopping immediately.")
            raise RuntimeError("Missing elites.json - evolution cannot continue. This indicates a critical system failure.")
        
        # Run adaptive parent selection (writes parents.json & top_10.json)
        self.parent_selector.adaptive_tournament_selection(evolution_tracker, outputs_path=str(self.outputs_path))
        
        # Validate that parents were selected by checking parents.json
        parents = self._load_parents_from_file()
        if not parents:
            self.logger.error("No parents selected or failed to load parents from file")
            return
        
        # EvolutionTracker is already updated by parent_selector.py when parents.json was created

        # Get operators (excluding InformedEvolution)
        single_parent_operators = self._get_single_parent_operators()
        multi_parent_operators = self._get_multi_parent_operators()
        
        # Run crossover phase first (multi-parent recombination for diversity)
        if len(parents) >= 2:
            self.logger.debug(f"Running crossover globally with {len(parents)} parents and {len(multi_parent_operators)} operators.")
            self._run_crossover_operators(parents, multi_parent_operators)
        
        # Run mutation phase (single-parent operations)
        if len(parents) >= 1:
            self.logger.debug(f"Running mutation globally with {len(parents)} parents and {len(single_parent_operators)} operators.")
            self._run_mutation_operators(parents, single_parent_operators)

    def _generate_variants_all_mode(self, evolution_tracker: Dict[str, Any] = None) -> None:
        """Generate variants using all operators with both parents.json and top_10.json"""
        self.logger.info("Running ALL mode: Using all operators with both parents.json and top_10.json")
        
        # Validate that elites exist before attempting parent selection
        elites_path = Path(self.outputs_path) / "elites.json"
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
            if not elites:
                self.logger.error("CRITICAL ERROR: elites.json exists but is empty - this indicates a fundamental problem")
                self.logger.error("Evolution cannot continue without elites. Stopping immediately.")
                raise RuntimeError("Empty elites.json - evolution cannot continue. This indicates a critical system failure.")
        else:
            self.logger.error("CRITICAL ERROR: elites.json does not exist - this indicates a fundamental problem")
            self.logger.error("Evolution cannot continue without elites. Stopping immediately.")
            raise RuntimeError("Missing elites.json - evolution cannot continue. This indicates a critical system failure.")
        
        # Run adaptive parent selection (writes parents.json & top_10.json)
        self.parent_selector.adaptive_tournament_selection(evolution_tracker, outputs_path=str(self.outputs_path))
        
        # Validate that parents were selected by checking parents.json
        parents = self._load_parents_from_file()
        if not parents:
            self.logger.error("No parents selected or failed to load parents from file")
            return
        
        # EvolutionTracker is already updated by parent_selector.py when parents.json was created

        # Get all operators
        single_parent_operators = self._get_single_parent_operators()
        multi_parent_operators = self._get_multi_parent_operators()
        
        # Run crossover phase first (multi-parent recombination for diversity)
        if len(parents) >= 2:
            self.logger.debug(f"Running crossover globally with {len(parents)} parents and {len(multi_parent_operators)} operators.")
            self._run_crossover_operators(parents, multi_parent_operators)
        
        # Run mutation phase (single-parent operations)
        if len(parents) >= 1:
            self.logger.debug(f"Running mutation globally with {len(parents)} parents and {len(single_parent_operators)} operators.")
            self._run_mutation_operators(parents, single_parent_operators)

    def _run_crossover_operators(self, parents: List[Dict], crossover_operators: List) -> None:
        """Run crossover operators on parent pairs"""
        for op in crossover_operators:
            if op.operator_type != "crossover":
                continue

            for parent_pair in combinations(parents, 2):  # All pairs of parents
                try:
                    # Call crossover operator max_variant times since it outputs one variant
                    variants_to_save = []
                    for _ in range(self.max_variants):
                        operator_input = {
                            "parent_data": list(parent_pair)
                        }
                        variants = op.apply(operator_input)
                        
                        if variants:
                            variants_to_save.extend([self._create_child_genome(vp, op, list(parent_pair), "crossover") for vp in variants])
                    
                    # Save variants immediately to temp.json
                    if variants_to_save:
                        self._append_variants_to_temp(variants_to_save)
                        self.logger.debug(f"Saved {len(variants_to_save)} crossover variants from {op.name}")
                        
                except Exception as e:
                    self.logger.error(f"[Crossover Error] {op.name} with parents {[p['id'] for p in parent_pair]}: {e}")

    def _run_mutation_operators(self, parents: List[Dict], mutation_operators: List) -> None:
        """Run mutation operators on parents"""
        for op in mutation_operators:
            if op.operator_type != "mutation":
                continue

            # Apply mutation to all parents max_variants times each
            for parent in parents:
                try:
                    variants_to_save = []
                    for variant_iteration in range(self.max_variants):
                        # Pass correct input type based on operator class
                        operator_input = {
                            "parent_data": parent
                        }
                        variants = op.apply(operator_input)
                        
                        # Collect variants
                        if variants:
                            variants_to_save.extend([self._create_child_genome(vp, op, [parent], "mutation") for vp in variants])
                    
                    # Save all variants to temp.json
                    if variants_to_save:
                        self._append_variants_to_temp(variants_to_save)
                        self.logger.debug(f"Saved {len(variants_to_save)} mutation variants from {op.name} for parent {parent['id']} ({self.max_variants} calls)")
                        
                except Exception as e:
                    self.logger.error(f"[Mutation Error] {op.name} with parent {parent['id']}: {e}")

        # Clean up parents and top_10 files after processing
        self.clean_parents_file()
    
    def _load_parents_from_file(self) -> List[Dict]:
        """
        Load parents from parents.json file.
        
        Returns:
            List[Dict]: List of parent genomes
        """
        try:
            parents_path = Path(self.outputs_path) / "parents.json"
            if not parents_path.exists():
                self.logger.warning("Parents file not found: %s", parents_path)
                return []
            
            with open(parents_path, 'r', encoding='utf-8') as f:
                parents_data = json.load(f)
            
            # Handle both old structure (with wrapper) and new structure (direct array)
            if isinstance(parents_data, list):
                # New structure: direct array
                parents = parents_data
            elif isinstance(parents_data, dict) and "parents" in parents_data:
                # Old structure: wrapper object
                parents = parents_data.get("parents", [])
            else:
                self.logger.warning("Unexpected parents.json structure")
                return []
            
            self.logger.debug(f"Loaded {len(parents)} parents from file: {[p['id'] for p in parents]}")
            
            return parents
            
        except Exception as e:
            self.logger.error(f"Failed to load parents from file: {e}")
            return []
    

    def _append_variants_to_temp(self, variants: List[Dict]) -> None:
        """
        Append variants to temp.json file.
        
        Args:
            variants: List of variant genomes to append
        """
        try:
            temp_path = Path(self.outputs_path) / "temp.json"
            
            # Load existing variants
            if temp_path.exists():
                with open(temp_path, 'r', encoding='utf-8') as f:
                    existing_variants = json.load(f)
            else:
                existing_variants = []
            
            # Append new variants
            existing_variants.extend(variants)
            
            # Save back to file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(existing_variants, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Appended {len(variants)} variants to temp.json (total: {len(existing_variants)})")
            
        except Exception as e:
            self.logger.error(f"Failed to append variants to temp.json: {e}")
            raise
    
    def clean_parents_file(self) -> None:
        """Read parents.json and top_10.json, update EvolutionTracker, then empty the files after all operators have processed the parents."""
        try: 
            parents_path = Path(self.outputs_path) / "parents.json"
            top10_path = Path(self.outputs_path) / "top_10.json"
            
            # Read files before cleaning and update EvolutionTracker
            self._update_evolution_tracker_from_files(parents_path, top10_path)
            
            # Now clean the files
            emptied = []
            for path in [parents_path, top10_path]:
                # Ensure parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                # Write empty list to file (creates or overwrites)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
                emptied.append(str(path))
            if emptied:
                self.logger.info(f"Emptied files: {', '.join(emptied)}")
        except Exception as e:
            self.logger.error(f"Failed to empty parents/top_10 file: {e}")

    def _update_evolution_tracker_from_files(self, parents_path: Path, top10_path: Path) -> None:
        """
        Read parents.json and top_10.json files and update EvolutionTracker with just the genome IDs.
        
        Args:
            parents_path: Path to parents.json file
            top10_path: Path to top_10.json file
        """
        try:
            import json
            from pathlib import Path
            from utils.population_io import get_outputs_path
            
            # Load EvolutionTracker
            evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
            if not evolution_tracker_path.exists():
                self.logger.warning("EvolutionTracker.json not found for update")
                return
            
            with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            
            # Get current generation number - use current_cycle directly
            current_generation = self.current_cycle
            if current_generation is None:
                self.logger.error("current_cycle is None - cannot determine generation number")
                return
            
            # Find or create the current generation entry
            current_gen = None
            for gen in tracker.get("generations", []):
                if gen.get("generation_number") == current_generation:
                    current_gen = gen
                    break
            
            if current_gen is None:
                # Create new generation entry
                current_gen = {
                    "generation_number": current_generation,
                    "genome_id": None,
                    "max_score": 0.0001,
                    "min_score": 0.0001,
                    "avg_fitness": 0.0001,
                    # Variant statistics from temp.json (before distribution)
                    "max_score_variants": 0.0001,
                    "min_score_variants": 0.0001,
                    "avg_fitness_variants": 0.0001,
                    # Population statistics (after distribution)
                    "avg_fitness_generation": 0.0001,
                    "avg_fitness_elites": 0.0001,
                    "avg_fitness_non_elites": 0.0001,
                    "parents": [],
                    "top_10": [],
                    "variants_created": 0,
                    "mutation_variants": 0,
                    "crossover_variants": 0,
                    "elites_threshold": 0.0001,
                    "removal_threshold": 0.0001,
                    "elites_count": 0,
                    "non_elites_count": 0
                }
                tracker.setdefault("generations", []).append(current_gen)
                self.logger.info(f"Created new generation entry: {current_generation}")
            
            # Read parent IDs from parents.json
            parent_ids = []
            if parents_path.exists():
                with open(parents_path, 'r', encoding='utf-8') as f:
                    parents_data = json.load(f)
                if isinstance(parents_data, list) and parents_data:
                    parent_ids = [str(p.get("id")) for p in parents_data if p.get("id")]
            
            # Read top_10 IDs from top_10.json
            top_10_ids = []
            if top10_path.exists():
                with open(top10_path, 'r', encoding='utf-8') as f:
                    top_10_data = json.load(f)
                if isinstance(top_10_data, list) and top_10_data:
                    top_10_ids = [str(genome.get("id")) for genome in top_10_data if genome and genome.get("id")]
            
            # Update the generation entry with just the IDs
            current_gen["parents"] = parent_ids
            current_gen["top_10"] = top_10_ids
            
            # Save updated EvolutionTracker
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(tracker, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Updated EvolutionTracker with {len(parent_ids)} parent IDs and {len(top_10_ids)} top_10 IDs for generation {current_generation}")
            
        except Exception as e:
            self.logger.error(f"Failed to update EvolutionTracker from files: {e}")

    def _get_single_parent_operators(self):
        """Return list of mutation operators that require only a single parent."""
        
        # Initialize operators based on configuration to avoid unnecessary initialization
        if self.operators == "ie":
            # Only InformedEvolution operator
            filtered_operators = [
                InformedEvolutionOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator, top_10_path=str(self.outputs_path / "top_10.json"))
            ]
            self.logger.info("Operator mode 'ie': Using only InformedEvolution operator (%d operators)", len(filtered_operators))
        elif self.operators == "cm":
            # All operators except InformedEvolution
            filtered_operators = [
                # LLM-based POS-aware operators
                LLM_POSAwareSynonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                POSAwareAntonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                
                # LLM-based text transformation operators
                MLMOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                LLMBasedParaphrasingOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                StylisticMutator(log_file=self.log_file, generator=self.prompt_generator),
                
                # Back translation operators
                LLMBackTranslationHIOperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationFROperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationDEOperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationJAOperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationZHOperator(log_file=self.log_file, generator=self.prompt_generator),
                
                # New mutation operators
                NegationOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                TypographicalErrorsOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                ConceptAdditionOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
            ]
            self.logger.info("Operator mode 'cm': Using all operators except InformedEvolution (%d operators)", len(filtered_operators))
        elif self.operators == "all":
            # All operators
            filtered_operators = [
                # LLM-based POS-aware operators
                LLM_POSAwareSynonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                POSAwareAntonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                
                # LLM-based text transformation operators
                MLMOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                LLMBasedParaphrasingOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                StylisticMutator(log_file=self.log_file, generator=self.prompt_generator),
                
                # Back translation operators
                LLMBackTranslationHIOperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationFROperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationDEOperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationJAOperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationZHOperator(log_file=self.log_file, generator=self.prompt_generator),
                
                # New mutation operators
                NegationOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                TypographicalErrorsOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                ConceptAdditionOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                InformedEvolutionOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator, top_10_path=str(self.outputs_path / "top_10.json")),
            ]
            self.logger.info("Operator mode 'all': Using all operators (%d operators)", len(filtered_operators))
        else:
            # Default to all operators if invalid mode
            filtered_operators = [
                # LLM-based POS-aware operators
                LLM_POSAwareSynonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                POSAwareAntonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                
                # LLM-based text transformation operators
                MLMOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                LLMBasedParaphrasingOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                StylisticMutator(log_file=self.log_file, generator=self.prompt_generator),
                
                # Back translation operators
                LLMBackTranslationHIOperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationFROperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationDEOperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationJAOperator(log_file=self.log_file, generator=self.prompt_generator),
                LLMBackTranslationZHOperator(log_file=self.log_file, generator=self.prompt_generator),
                
                # New mutation operators
                NegationOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                TypographicalErrorsOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                ConceptAdditionOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                InformedEvolutionOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator, top_10_path=str(self.outputs_path / "top_10.json")),
            ]
            self.logger.warning("Invalid operator mode '%s', defaulting to 'all' (%d operators)", self.operators, len(filtered_operators))
        
        return filtered_operators

    def _get_multi_parent_operators(self):
        """Return list of crossover operators that require multiple parents."""
        
        # Initialize operators based on configuration to avoid unnecessary initialization
        if self.operators == "ie":
            # Only InformedEvolution operator - no crossover operators
            filtered_operators = []
            self.logger.info("Operator mode 'ie': No crossover operators enabled")
        elif self.operators == "cm":
            # All crossover operators (no InformedEvolution in crossover)
            filtered_operators = [
                SemanticSimilarityCrossover(log_file=self.log_file),
                SemanticFusionCrossover(north_star_metric=self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator)
            ]
            self.logger.info("Operator mode 'cm': Using all crossover operators (%d operators)", len(filtered_operators))
        elif self.operators == "all":
            # All crossover operators
            filtered_operators = [
                SemanticSimilarityCrossover(log_file=self.log_file),
                SemanticFusionCrossover(north_star_metric=self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator)
            ]
            self.logger.info("Operator mode 'all': Using all crossover operators (%d operators)", len(filtered_operators))
        else:
            # Default to all operators if invalid mode
            filtered_operators = [
                SemanticSimilarityCrossover(log_file=self.log_file),
                SemanticFusionCrossover(north_star_metric=self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator)
            ]
            self.logger.warning("Invalid operator mode '%s', defaulting to 'all' for crossover (%d operators)", self.operators, len(filtered_operators))
        
        return filtered_operators

    def get_last_generation_data(self) -> Dict[str, Any]:
        """Get the generation data from the last run for tracking purposes."""
        return getattr(self, '_last_generation_data', {
            "generation_number": self.current_cycle,
            "parents": [],
            "variants_created": 0,
            "mutation_variants": 0,
            "crossover_variants": 0
        })

    def _deduplicate_temp_json(self) -> int:
        """
        Remove duplicate variants within temp.json based on normalized prompt.
        Keeps the first occurrence and discards subsequent duplicates.
        
        Returns:
            int: Number of duplicates removed
        """
        try:
            temp_path = Path(self.outputs_path) / "temp.json"
            if not temp_path.exists():
                return 0

            with open(temp_path, 'r', encoding='utf-8') as f:
                variants = json.load(f)

            if not isinstance(variants, list) or not variants:
                return 0

            seen_prompts = set()
            seen_ids = set()
            unique_variants = []
            duplicates_removed = 0

            for v in variants:
                if not isinstance(v, dict):
                    # Skip invalid entries but count as removed when rewriting list
                    duplicates_removed += 1
                    continue

                prompt = v.get("prompt")
                vid = v.get("id")

                # Normalize prompt for dedup
                norm = prompt.strip().lower() if isinstance(prompt, str) else None

                # Criteria: duplicate if same normalized prompt OR duplicate id already seen
                if (norm is not None and norm in seen_prompts) or (vid is not None and vid in seen_ids):
                    duplicates_removed += 1
                    continue

                if norm is not None:
                    seen_prompts.add(norm)
                if vid is not None:
                    seen_ids.add(vid)
                unique_variants.append(v)

            # Only rewrite file if duplicates were removed
            if duplicates_removed > 0:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(unique_variants, f, indent=2, ensure_ascii=False)

            return duplicates_removed
        except Exception as e:
            self.logger.error(f"Failed to deduplicate temp.json: {e}")
            return 0

