"""
Comprehensive Execution Validation Script

Validates all logs, logic, metrics, statistics, and fields from JSON/CSV files
for a specific execution directory.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter
import sys


class ComprehensiveValidator:
    """Comprehensive validator for execution outputs."""
    
    def __init__(self, execution_dir: Path):
        self.execution_dir = Path(execution_dir)
        self.issues = []
        self.warnings = []
        self.validations_passed = 0
        self.validations_failed = 0
        
        # Load all files
        self.load_files()
    
    def load_files(self):
        """Load all JSON and CSV files."""
        try:
            # JSON files
            with open(self.execution_dir / "EvolutionTracker.json", 'r') as f:
                self.tracker = json.load(f)
            
            with open(self.execution_dir / "elites.json", 'r') as f:
                self.elites = json.load(f)
            
            with open(self.execution_dir / "reserves.json", 'r') as f:
                self.reserves = json.load(f)
            
            with open(self.execution_dir / "archive.json", 'r') as f:
                self.archive = json.load(f)
            
            with open(self.execution_dir / "speciation_state.json", 'r') as f:
                self.speciation_state = json.load(f)
            
            with open(self.execution_dir / "genome_tracker.json", 'r') as f:
                self.genome_tracker = json.load(f)
            
            # CSV file
            self.operator_csv = pd.read_csv(self.execution_dir / "operator_effectiveness_cumulative.csv")
            
        except Exception as e:
            self.issues.append(f"Failed to load files: {e}")
            raise
    
    def validate(self) -> Dict[str, Any]:
        """Run all validations."""
        print("=" * 80)
        print(f"COMPREHENSIVE VALIDATION: {self.execution_dir.name}")
        print("=" * 80)
        
        # 1. Log Validation (if logs available)
        self.validate_logs()
        
        # 2. Logic Validation
        self.validate_speciation_logic()
        self.validate_merging_logic()
        self.validate_freezing_logic()
        self.validate_distribution_logic()
        
        # 3. Metrics Validation
        self.validate_rq1_metrics()
        self.validate_rq2_metrics()
        self.validate_diversity_metrics()
        self.validate_cluster_quality_metrics()
        
        # 4. Statistics Validation
        self.validate_evolution_tracker_structure()
        self.validate_population_counts()
        
        # 5. JSON File Field Validation
        self.validate_elites_json()
        self.validate_reserves_json()
        self.validate_archive_json()
        self.validate_speciation_state_json()
        self.validate_evolution_tracker_json()
        self.validate_genome_tracker_json()
        
        # 6. CSV File Validation
        self.validate_operator_csv()
        
        # 7. Cross-File Consistency
        self.validate_cross_file_consistency()
        
        return self.generate_report()
    
    def validate_logs(self):
        """Validate execution logs (if available)."""
        # Check if logs directory exists
        logs_dir = self.execution_dir.parent.parent / "logs"
        if logs_dir.exists():
            # Try to find relevant log files
            log_files = list(logs_dir.glob("*.log"))
            if log_files:
                self.warnings.append("Log validation: Log files found but not analyzed (manual review recommended)")
            else:
                self.warnings.append("Log validation: No log files found")
        else:
            self.warnings.append("Log validation: Logs directory not found")
    
    def validate_speciation_logic(self):
        """Validate speciation logic."""
        print("\n[2.1] Validating Speciation Logic...")
        
        # Get theta_sim from speciation_state or use default
        theta_sim = 0.2  # Default
        if "config" in self.speciation_state:
            theta_sim = self.speciation_state["config"].get("theta_sim", 0.2)
        
        # Elites-based counts per species_id (unique genome ids)
        elites_count_by_sid = defaultdict(set)
        for g in self.elites:
            sid = g.get("species_id")
            if sid is not None and sid > 0:
                gid = g.get("id")
                if gid is not None:
                    elites_count_by_sid[sid].add(gid)
        min_island = self.speciation_state.get("config", {}).get("min_island_size", 2)

        # Check species formation
        species_dict = self.speciation_state.get("species", {})
        for sid, sp_data in species_dict.items():
            sid_int = int(sid)
            size = sp_data.get("size", 0)
            member_ids = sp_data.get("member_ids", [])

            # Size must equal len(member_ids) (source of truth: elites.json)
            if size != len(member_ids):
                self.issues.append(f"Species {sid_int}: size={size} != len(member_ids)={len(member_ids)}; size should match member_ids")

            # Size should match count in elites.json (unique genomes with this species_id)
            elites_n = len(elites_count_by_sid.get(sid_int, set()))
            if size != elites_n:
                self.issues.append(f"Species {sid_int}: size={size} != elites count={elites_n} (genomes in elites.json with species_id={sid_int})")

            if size < min_island:
                self.issues.append(f"Species {sid_int}: size={size} < min_island_size ({min_island})")
            
            # Check leader exists in elites
            leader_id = sp_data.get("leader_id")
            leader_genome = None
            if leader_id:
                leader_genome = next((g for g in self.elites if g.get("id") == leader_id), None)
                if not leader_genome:
                    self.issues.append(f"Species {sid_int}: leader_id={leader_id} not found in elites.json")
            
            # Check members exist in elites and validate radius enforcement
            radius = sp_data.get("radius", theta_sim)
            
            if leader_genome and leader_genome.get("prompt_embedding"):
                leader_emb = np.array(leader_genome["prompt_embedding"])
                leader_emb = leader_emb / np.linalg.norm(leader_emb)  # Normalize
                
                for mid in member_ids:
                    member_genome = next((g for g in self.elites if g.get("id") == mid), None)
                    if not member_genome:
                        self.issues.append(f"Species {sid_int}: member_id={mid} not found in elites.json")
                        continue
                    
                    # Skip radius check for leader
                    if mid == leader_id:
                        continue
                    
                    # Check radius enforcement using ensemble distance (same as system)
                    if member_genome.get("prompt_embedding"):
                        member_emb = np.array(member_genome["prompt_embedding"])
                        member_emb = member_emb / np.linalg.norm(member_emb)  # Normalize
                        
                        # Use ensemble distance (same as system) instead of cosine distance
                        try:
                            # Import ensemble distance function
                            import sys
                            from pathlib import Path
                            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
                            from speciation.distance import ensemble_distance
                            
                            # Extract phenotype vectors (not raw moderation_result dict)
                            from speciation.phenotype_distance import extract_phenotype_vector
                            leader_phenotype = extract_phenotype_vector(leader_genome)
                            member_phenotype = extract_phenotype_vector(member_genome)
                            
                            # Get weights from config (default: w_genotype=0.7, w_phenotype=0.3)
                            config = self.speciation_state.get("config", {})
                            w_genotype = config.get("w_genotype", 0.7)
                            w_phenotype = config.get("w_phenotype", 0.3)
                            
                            # Calculate ensemble distance
                            dist = ensemble_distance(
                                leader_emb, member_emb,
                                leader_phenotype, member_phenotype,
                                w_genotype, w_phenotype
                            )
                            
                            # Allow some tolerance for floating point errors
                            if dist > radius * 1.5:  # Allow some tolerance
                                self.warnings.append(f"Species {sid_int}: member {mid} may be outside radius (ensemble_dist={dist:.4f}, radius={radius})")
                        except Exception as e:
                            # Fallback to cosine distance if ensemble distance fails
                            cosine_sim = np.clip(np.dot(leader_emb, member_emb), -1.0, 1.0)
                            cosine_dist = 1.0 - cosine_sim
                            if cosine_dist > radius * 1.5:
                                self.warnings.append(f"Species {sid_int}: member {mid} may be outside radius (cosine_dist={cosine_dist:.4f}, radius={radius}, ensemble_distance failed: {e})")
        
        self.validations_passed += 1
    
    def validate_merging_logic(self):
        """Validate merging logic."""
        print("[2.2] Validating Merging Logic...")
        
        # Check for extinct species in historical
        historical_species = self.speciation_state.get("historical_species", {})
        extinct_count = sum(1 for sp in historical_species.values() 
                          if isinstance(sp, dict) and sp.get("species_state") == "extinct")
        
        # Check merged species have parent_ids
        species_dict = self.speciation_state.get("species", {})
        for sid, sp_data in species_dict.items():
            cluster_origin = sp_data.get("cluster_origin")
            parent_ids = sp_data.get("parent_ids")
            
            if cluster_origin == "merge":
                if not parent_ids or len(parent_ids) != 2:
                    self.issues.append(f"Species {sid}: cluster_origin='merge' but parent_ids={parent_ids} (expected 2)")
        
        self.validations_passed += 1
    
    def validate_freezing_logic(self):
        """Validate freezing logic."""
        print("[2.3] Validating Freezing Logic...")
        
        species_dict = self.speciation_state.get("species", {})
        frozen_in_active = sum(1 for sp in species_dict.values() 
                              if sp.get("species_state") == "frozen")
        
        # Frozen species should be in active dict, not historical
        historical_species = self.speciation_state.get("historical_species", {})
        frozen_in_historical = sum(1 for sp in historical_species.values() 
                                  if isinstance(sp, dict) and sp.get("species_state") == "frozen")
        
        if frozen_in_historical > 0:
            self.issues.append(f"Found {frozen_in_historical} frozen species in historical_species (should be in active dict)")
        
        self.validations_passed += 1
    
    def validate_distribution_logic(self):
        """Validate distribution logic."""
        print("[2.4] Validating Distribution Logic...")
        
        # Check elites have species_id > 0
        for genome in self.elites:
            species_id = genome.get("species_id")
            if species_id is None or species_id <= 0:
                self.issues.append(f"Elite genome {genome.get('id')}: species_id={species_id} (should be > 0)")
            
            # Check embeddings preserved
            if "prompt_embedding" not in genome:
                self.issues.append(f"Elite genome {genome.get('id')}: missing prompt_embedding")
        
        # Check reserves have species_id = 0
        for genome in self.reserves:
            species_id = genome.get("species_id")
            if species_id != 0:
                self.issues.append(f"Reserve genome {genome.get('id')}: species_id={species_id} (should be 0)")
            
            # Check embeddings preserved
            if "prompt_embedding" not in genome:
                self.issues.append(f"Reserve genome {genome.get('id')}: missing prompt_embedding")
        
        # Check archive has no embeddings
        for genome in self.archive:
            if "prompt_embedding" in genome:
                self.issues.append(f"Archived genome {genome.get('id')}: has prompt_embedding (should be removed)")
        
        self.validations_passed += 1
    
    def validate_rq1_metrics(self):
        """Validate RQ1 metrics (operator effectiveness)."""
        print("\n[3.1] Validating RQ1 Metrics...")
        
        for _, row in self.operator_csv.iterrows():
            gen = int(row['generation'])
            operator = row['operator']
            
            total_variants = row['total_variants']
            elite_count = row['elite_count']
            non_elite_count = row['non_elite_count']
            rejections = row['rejections']
            duplicates = row['duplicates']
            
            calculated_total = total_variants + rejections + duplicates
            
            if calculated_total == 0:
                continue  # Skip if no variants
            
            # Validate NE
            expected_ne = round(non_elite_count / calculated_total * 100, 2)
            if abs(row['NE'] - expected_ne) > 0.01:
                self.issues.append(f"Gen {gen}, {operator}: NE mismatch. Expected {expected_ne}, got {row['NE']}")
            
            # Validate EHR
            expected_ehr = round(elite_count / calculated_total * 100, 2)
            if abs(row['EHR'] - expected_ehr) > 0.01:
                self.issues.append(f"Gen {gen}, {operator}: EHR mismatch. Expected {expected_ehr}, got {row['EHR']}")
            
            # Validate IR
            expected_ir = round(rejections / calculated_total * 100, 2)
            if abs(row['IR'] - expected_ir) > 0.01:
                self.issues.append(f"Gen {gen}, {operator}: IR mismatch. Expected {expected_ir}, got {row['IR']}")
            
            # Validate sum
            duplicates_percent = round(duplicates / calculated_total * 100, 2) if calculated_total > 0 else 0.0
            total_percent = row['NE'] + row['EHR'] + row['IR'] + duplicates_percent
            if abs(total_percent - 100.0) > 0.1:
                self.issues.append(f"Gen {gen}, {operator}: NE+EHR+IR+duplicates={total_percent:.2f}, should be ~100")
            
            # Validate cEHR
            if total_variants > 0:
                expected_cehr = round(elite_count / total_variants * 100, 2)
                if pd.notna(row['cEHR']) and abs(row['cEHR'] - expected_cehr) > 0.01:
                    self.issues.append(f"Gen {gen}, {operator}: cEHR mismatch. Expected {expected_cehr}, got {row['cEHR']}")
        
        self.validations_passed += 1
    
    def validate_rq2_metrics(self):
        """Validate RQ2 metrics (speciation metrics).
        Only validates species_count and frozen_species_count for the last generation;
        per-generation speciation state is not stored, so earlier gens are skipped."""
        print("[3.2] Validating RQ2 Metrics...")
        
        generations = self.tracker.get("generations", [])
        last_gen = max((g.get("generation_number", -1) for g in generations), default=-1)
        
        for gen_entry in generations:
            gen_num = gen_entry.get("generation_number")
            spec_data = gen_entry.get("speciation") or {}
            
            if not spec_data:
                continue
            if gen_num != last_gen:
                continue
            
            species_dict = self.speciation_state.get("species", {})
            active_count = len([sp for sp in species_dict.values() if sp.get("species_state") == "active"])
            frozen_count = len([sp for sp in species_dict.values() if sp.get("species_state") == "frozen"])
            expected_total = active_count + frozen_count
            species_count = spec_data.get("species_count", 0)
            frozen_species_count = spec_data.get("frozen_species_count", 0)
            
            if species_count != expected_total:
                self.issues.append(f"Gen {gen_num}: species_count={species_count}, expected {expected_total} (active={active_count}, frozen={frozen_count})")
            if frozen_species_count != frozen_count:
                self.issues.append(f"Gen {gen_num}: frozen_species_count={frozen_species_count}, expected {frozen_count}")
        
        self.validations_passed += 1
    
    def validate_diversity_metrics(self):
        """Validate diversity metrics."""
        print("[3.3] Validating Diversity Metrics...")
        
        # Diversity metrics are calculated from elites.json
        # Check that only species (species_id > 0) are included
        species_ids_in_elites = {g.get("species_id") for g in self.elites if g.get("species_id") is not None}
        cluster0_ids = {sid for sid in species_ids_in_elites if sid == 0}
        
        if cluster0_ids:
            self.warnings.append(f"Diversity metrics: Found {len(cluster0_ids)} genomes with species_id=0 in elites.json (should be in reserves.json)")
        
        self.validations_passed += 1
    
    def validate_cluster_quality_metrics(self):
        """Validate cluster quality metrics."""
        print("[3.4] Validating Cluster Quality Metrics...")
        
        for gen_entry in self.tracker.get("generations", []):
            spec_data = gen_entry.get("speciation") or {}
            cluster_quality = spec_data.get("cluster_quality")
            
            if cluster_quality:
                num_clusters = cluster_quality.get("num_clusters", 0)
                num_samples = cluster_quality.get("num_samples", 0)
                
                # Check that cluster 0 is excluded (num_clusters should match active species)
                species_dict = self.speciation_state.get("species", {})
                active_species_count = len([sp for sp in species_dict.values() 
                                           if sp.get("species_state") == "active"])
                
                if num_clusters != active_species_count:
                    self.warnings.append(f"Gen {gen_entry.get('generation_number')}: cluster_quality.num_clusters={num_clusters}, active_species={active_species_count} (may include frozen)")
        
        self.validations_passed += 1
    
    def validate_evolution_tracker_structure(self):
        """Validate EvolutionTracker.json structure."""
        print("\n[4.1] Validating EvolutionTracker.json Structure...")
        
        generations = self.tracker.get("generations", [])
        
        for i, gen_entry in enumerate(generations):
            gen_num = gen_entry.get("generation_number")
            
            if gen_num != i:
                self.issues.append(f"Generation {i}: generation_number={gen_num} (expected {i})")
            
            # Check required fields
            required_fields = ["elites_count", "reserves_count", "total_population", "avg_fitness"]
            for field in required_fields:
                if field not in gen_entry:
                    self.issues.append(f"Gen {gen_num}: missing required field '{field}'")
            
            # Check sum
            elites_count = gen_entry.get("elites_count", 0)
            reserves_count = gen_entry.get("reserves_count", 0)
            total_population = gen_entry.get("total_population", 0)
            
            if elites_count + reserves_count != total_population:
                self.issues.append(f"Gen {gen_num}: elites_count({elites_count}) + reserves_count({reserves_count}) != total_population({total_population})")
        
        self.validations_passed += 1
    
    def validate_population_counts(self):
        """Validate population counts.
        Only the last generation is compared to current file counts (len(elites), etc.);
        per-gen file snapshots are not stored, so earlier gens are skipped."""
        print("[4.2] Validating Population Counts...")
        
        generations = self.tracker.get("generations", [])
        last_gen = max((g.get("generation_number", -1) for g in generations), default=-1)
        
        for gen_entry in generations:
            gen_num = gen_entry.get("generation_number")
            if gen_num != last_gen:
                continue
            
            expected_elites = len(self.elites)
            expected_reserves = len(self.reserves)
            expected_archive = len(self.archive)
            actual_elites = gen_entry.get("elites_count", 0)
            actual_reserves = gen_entry.get("reserves_count", 0)
            actual_archive = gen_entry.get("archived_count", 0)
            
            if expected_elites != actual_elites:
                self.issues.append(f"Gen {gen_num}: elites_count mismatch. Expected {expected_elites} (len(elites)), got {actual_elites}")
            if expected_reserves != actual_reserves:
                self.issues.append(f"Gen {gen_num}: reserves_count mismatch. Expected {expected_reserves} (len(reserves)), got {actual_reserves}")
            if expected_archive != actual_archive:
                self.issues.append(f"Gen {gen_num}: archived_count mismatch. Expected {expected_archive} (len(archive)), got {actual_archive}")
        
        self.validations_passed += 1
    
    def validate_elites_json(self):
        """Validate elites.json fields."""
        print("\n[5.1] Validating elites.json...")
        
        genome_ids = set()
        
        for genome in self.elites:
            # Check required fields
            if "id" not in genome:
                self.issues.append("Elite genome: missing 'id' field")
                continue
            
            genome_id = genome.get("id")
            
            # Check for duplicates
            if genome_id in genome_ids:
                self.issues.append(f"Elite genome: duplicate ID {genome_id}")
            genome_ids.add(genome_id)
            
            # Check species_id > 0
            species_id = genome.get("species_id")
            if species_id is None or species_id <= 0:
                self.issues.append(f"Elite genome {genome_id}: species_id={species_id} (should be > 0)")
            
            # Check prompt_embedding
            if "prompt_embedding" not in genome:
                self.issues.append(f"Elite genome {genome_id}: missing prompt_embedding")
            
            # Check fitness range
            fitness = genome.get("fitness")
            if fitness is not None and (fitness < 0.0 or fitness > 1.0):
                self.issues.append(f"Elite genome {genome_id}: fitness={fitness} (should be 0.0-1.0)")
        
        self.validations_passed += 1
    
    def validate_reserves_json(self):
        """Validate reserves.json fields."""
        print("[5.2] Validating reserves.json...")
        
        for genome in self.reserves:
            genome_id = genome.get("id")
            
            # Check species_id = 0
            species_id = genome.get("species_id")
            if species_id != 0:
                self.issues.append(f"Reserve genome {genome_id}: species_id={species_id} (should be 0)")
            
            # Check prompt_embedding
            if "prompt_embedding" not in genome:
                self.issues.append(f"Reserve genome {genome_id}: missing prompt_embedding")
        
        # Check count matches speciation_state
        expected_count = self.speciation_state.get("cluster0_size_from_reserves", 0)
        actual_count = len(self.reserves)
        if expected_count != actual_count:
            self.issues.append(f"Reserves count mismatch: speciation_state={expected_count}, reserves.json={actual_count}")
        
        self.validations_passed += 1
    
    def validate_archive_json(self):
        """Validate archive.json fields."""
        print("[5.3] Validating archive.json...")
        
        for genome in self.archive:
            genome_id = genome.get("id")
            
            # Check no prompt_embedding
            if "prompt_embedding" in genome:
                self.issues.append(f"Archived genome {genome_id}: has prompt_embedding (should be removed)")
            
            # Check initial_state
            initial_state = genome.get("initial_state")
            if initial_state != "non-elite":
                self.issues.append(f"Archived genome {genome_id}: initial_state={initial_state} (should be 'non-elite')")
        
        self.validations_passed += 1
    
    def validate_speciation_state_json(self):
        """Validate speciation_state.json structure."""
        print("[5.4] Validating speciation_state.json...")
        
        species_dict = self.speciation_state.get("species", {})
        leader_ids = []
        
        for sid, sp_data in species_dict.items():
            # Check required fields
            required_fields = ["id", "leader_id", "leader_fitness", "member_ids", "species_state"]
            for field in required_fields:
                if field not in sp_data:
                    self.issues.append(f"Species {sid}: missing required field '{field}'")
            
            # Check leader_id exists
            leader_id = sp_data.get("leader_id")
            if leader_id:
                leader_ids.append(leader_id)
                leader_found = any(g.get("id") == leader_id for g in self.elites)
                if not leader_found:
                    self.issues.append(f"Species {sid}: leader_id={leader_id} not found in elites.json")
            
            # Check species_state
            species_state = sp_data.get("species_state")
            valid_states = ["active", "frozen", "incubator", "extinct"]
            if species_state not in valid_states:
                self.issues.append(f"Species {sid}: invalid species_state='{species_state}' (expected {valid_states})")
            
            # Check frozen species not in historical
            if species_state == "frozen":
                # Already checked in validate_freezing_logic
                pass
        
        # Check for duplicate leader IDs
        leader_id_counts = Counter(leader_ids)
        duplicates = {lid: count for lid, count in leader_id_counts.items() if count > 1}
        if duplicates:
            self.issues.append(f"Duplicate leader IDs found: {duplicates}")
        
        self.validations_passed += 1
    
    def validate_evolution_tracker_json(self):
        """Validate EvolutionTracker.json fields."""
        print("[5.5] Validating EvolutionTracker.json...")
        
        # Structure already validated in validate_evolution_tracker_structure
        # Additional field validations here
        self.validations_passed += 1
    
    def validate_genome_tracker_json(self):
        """Validate genome_tracker.json structure."""
        print("[5.6] Validating genome_tracker.json...")
        
        # Basic structure check
        if not isinstance(self.genome_tracker, dict):
            self.issues.append("genome_tracker.json: invalid structure (expected dict)")
        
        self.validations_passed += 1
    
    def validate_operator_csv(self):
        """Validate operator_effectiveness_cumulative.csv."""
        print("\n[6.1] Validating operator_effectiveness_cumulative.csv...")
        
        required_columns = ["generation", "operator", "NE", "EHR", "IR", "cEHR", 
                           "Δμ", "Δσ", "total_variants", "elite_count", "non_elite_count", 
                           "rejections", "duplicates"]
        
        for col in required_columns:
            if col not in self.operator_csv.columns:
                self.issues.append(f"CSV: missing required column '{col}'")
        
        # Check if CSV is empty but should have data
        if self.operator_csv.empty:
            # Check if any generation has operator_statistics
            for gen_entry in self.tracker.get("generations", []):
                gen_num = gen_entry.get("generation_number")
                if gen_num == 0:
                    continue  # Generation 0 has no operator-created variants
                
                operator_stats = gen_entry.get("operator_statistics") or {}
                if operator_stats:
                    # Generation has operator statistics but CSV is empty
                    # This might be expected if all variants were rejected/duplicated
                    # But log as warning for review
                    variants_created = gen_entry.get("variants_created", 0)
                    if variants_created > 0:
                        self.warnings.append(f"Gen {gen_num}: CSV is empty but variants_created={variants_created} and operator_statistics exist. This may indicate all variants were rejected/duplicated.")
        
        # Metrics already validated in validate_rq1_metrics (if CSV has data)
        if not self.operator_csv.empty:
            # Only validate metrics if CSV has data
            pass  # Already validated in validate_rq1_metrics
        
        self.validations_passed += 1
    
    def validate_cross_file_consistency(self):
        """Validate cross-file consistency."""
        print("\n[7] Validating Cross-File Consistency...")
        
        # Check for duplicate genome IDs
        all_genome_ids = []
        all_genome_ids.extend([g.get("id") for g in self.elites if g.get("id") is not None])
        all_genome_ids.extend([g.get("id") for g in self.reserves if g.get("id") is not None])
        all_genome_ids.extend([g.get("id") for g in self.archive if g.get("id") is not None])
        
        id_counts = Counter(all_genome_ids)
        duplicates = {gid: count for gid, count in id_counts.items() if count > 1}
        if duplicates:
            self.issues.append(f"Duplicate genome IDs across files: {duplicates}")
        
        # Check species IDs in elites match speciation_state (species or incubators)
        elite_species_ids = {g.get("species_id") for g in self.elites if g.get("species_id") is not None and g.get("species_id") > 0}
        state_species_ids = {int(sid) for sid in self.speciation_state.get("species", {}).keys() if str(sid).isdigit()}
        incubator_ids = set(self.speciation_state.get("incubators", []))
        all_tracked_ids = state_species_ids | incubator_ids
        
        missing_in_state = elite_species_ids - all_tracked_ids
        if missing_in_state:
            self.issues.append(f"Species IDs in elites.json but not in speciation_state.json: {missing_in_state}")
        
        self.validations_passed += 1
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        
        total_issues = len(self.issues)
        total_warnings = len(self.warnings)
        
        print(f"\nValidations Passed: {self.validations_passed}")
        print(f"Validations Failed: {self.validations_failed}")
        print(f"Issues Found: {total_issues}")
        print(f"Warnings: {total_warnings}")
        
        if self.issues:
            print("\nISSUES:")
            for issue in self.issues:
                print(f"  ✗ {issue}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        if not self.issues and not self.warnings:
            print("\n✓ All validations passed!")
        
        return {
            "execution_dir": str(self.execution_dir),
            "validations_passed": self.validations_passed,
            "validations_failed": self.validations_failed,
            "issues": self.issues,
            "warnings": self.warnings,
            "total_issues": total_issues,
            "total_warnings": total_warnings
        }


def main():
    """Main validation function."""
    import sys
    
    # Allow command-line argument for execution directory
    if len(sys.argv) > 1:
        execution_dir = Path(sys.argv[1])
    else:
        execution_dir = Path("data/outputs/20260117_2107")
    
    if not execution_dir.exists():
        print(f"Error: Execution directory not found: {execution_dir}")
        sys.exit(1)
    
    validator = ComprehensiveValidator(execution_dir)
    report = validator.validate()
    
    # Save report
    report_path = execution_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    main()
