#!/usr/bin/env python3
"""
Comprehensive validation script for all output files, fields, metrics, and statistics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from utils.validate_evolution_tracker import validate_evolution_tracker_comprehensive
    HAS_ET_VALIDATOR = True
except ImportError:
    HAS_ET_VALIDATOR = False

try:
    from speciation.validation import validate_speciation_consistency
    HAS_SPE_VALIDATOR = True
except ImportError:
    HAS_SPE_VALIDATOR = False

try:
    from utils import get_custom_logging, get_system_utils
    get_logger, _, _, _ = get_custom_logging()
    _, _, _, get_outputs_path, _, _ = get_system_utils()
except ImportError:
    # Fallback logger
    import logging
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)
    get_outputs_path = lambda: Path("data/outputs")


def load_json_file(file_path: Path) -> Any:
    """Load JSON file safely."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def validate_file_structure(outputs_path: Path, logger) -> Dict[str, Any]:
    """Validate that all required files exist and are valid JSON."""
    results = {"valid": True, "errors": [], "warnings": []}
    
    required_files = [
        "EvolutionTracker.json",
        "elites.json",
        "reserves.json",
        "archive.json",
        "speciation_state.json",
        "genome_tracker.json",
        "events_tracker.json"
    ]
    
    optional_files = [
        "temp.json",
        "parents.json",
        "top_10.json",
        "operator_effectiveness_cumulative.csv"
    ]
    
    for filename in required_files:
        file_path = outputs_path / filename
        if not file_path.exists():
            results["errors"].append(f"Required file missing: {filename}")
            results["valid"] = False
        else:
            data = load_json_file(file_path)
            if isinstance(data, dict) and "error" in data:
                results["errors"].append(f"Failed to load {filename}: {data['error']}")
                results["valid"] = False
    
    for filename in optional_files:
        file_path = outputs_path / filename
        if file_path.exists():
            data = load_json_file(file_path)
            if isinstance(data, dict) and "error" in data:
                results["warnings"].append(f"Failed to load optional {filename}: {data['error']}")
    
    return results


def validate_genome_tracker(outputs_path: Path, logger) -> Dict[str, Any]:
    """Validate genome_tracker.json structure and consistency."""
    results = {"valid": True, "errors": [], "warnings": []}
    
    tracker_path = outputs_path / "genome_tracker.json"
    if not tracker_path.exists():
        results["errors"].append("genome_tracker.json not found")
        results["valid"] = False
        return results
    
    tracker = load_json_file(tracker_path)
    if isinstance(tracker, dict) and "error" in tracker:
        results["errors"].append(f"Failed to load genome_tracker.json: {tracker['error']}")
        results["valid"] = False
        return results
    
    # Check structure
    if not isinstance(tracker, dict):
        results["errors"].append("genome_tracker.json is not a dictionary")
        results["valid"] = False
        return results
    
    if "genomes" not in tracker:
        results["errors"].append("genome_tracker.json missing 'genomes' field")
        results["valid"] = False
    else:
        genomes = tracker.get("genomes", {})
        if not isinstance(genomes, dict):
            results["errors"].append("genome_tracker.genomes is not a dictionary")
            results["valid"] = False
        else:
            # Check that all species_id values are valid
            invalid_species_ids = []
            for genome_id, genome_data in genomes.items():
                species_id = genome_data.get("species_id")
                if species_id is not None:
                    if not isinstance(species_id, int):
                        invalid_species_ids.append(f"{genome_id}: species_id={species_id} (not int)")
                    elif species_id < -1:
                        invalid_species_ids.append(f"{genome_id}: species_id={species_id} (must be >= -1)")
            
            if invalid_species_ids:
                results["errors"].extend(invalid_species_ids[:10])  # Limit to first 10
                results["valid"] = False
    
    return results


def validate_distribution_rules(outputs_path: Path, logger) -> Dict[str, Any]:
    """Validate distribution rules: species_id >0→elites, 0→reserves, -1→archive."""
    results = {"valid": True, "errors": []}
    
    elites = load_json_file(outputs_path / "elites.json") or []
    reserves = load_json_file(outputs_path / "reserves.json") or []
    archive = load_json_file(outputs_path / "archive.json") or []
    
    # Ensure lists
    if not isinstance(elites, list):
        elites = []
    if not isinstance(reserves, list):
        reserves = []
    if not isinstance(archive, list):
        archive = []
    
    # Check elites: must have species_id > 0
    for g in elites:
        sid = g.get("species_id")
        if sid is None or sid <= 0:
            results["errors"].append(f"Elite genome id={g.get('id')} has species_id={sid} (must be > 0)")
            results["valid"] = False
    
    # Check reserves: must have species_id = 0
    for g in reserves:
        sid = g.get("species_id")
        if sid is not None and sid != 0:
            results["errors"].append(f"Reserve genome id={g.get('id')} has species_id={sid} (must be 0)")
            results["valid"] = False
    
    # Check archive: must have species_id = -1
    for g in archive:
        sid = g.get("species_id")
        if sid is not None and sid != -1:
            results["errors"].append(f"Archive genome id={g.get('id')} has species_id={sid} (must be -1)")
            results["valid"] = False
    
    return results


def validate_population_counts(outputs_path: Path, logger) -> Dict[str, Any]:
    """Validate population counts across files and EvolutionTracker."""
    results = {"valid": True, "errors": [], "warnings": []}
    
    elites = load_json_file(outputs_path / "elites.json") or []
    reserves = load_json_file(outputs_path / "reserves.json") or []
    archive = load_json_file(outputs_path / "archive.json") or []
    tracker = load_json_file(outputs_path / "EvolutionTracker.json")
    
    if not isinstance(elites, list):
        elites = []
    if not isinstance(reserves, list):
        reserves = []
    if not isinstance(archive, list):
        archive = []
    
    actual_elites_count = len(elites)
    actual_reserves_count = len(reserves)
    actual_archive_count = len(archive)
    actual_total = actual_elites_count + actual_reserves_count
    
    if tracker and isinstance(tracker, dict):
        generations = tracker.get("generations", [])
        if generations:
            latest_gen = max(generations, key=lambda g: g.get("generation_number", -1))
            
            # Check counts
            et_elites = latest_gen.get("elites_count", 0)
            et_reserves = latest_gen.get("reserves_count", 0)
            et_total = latest_gen.get("total_population", 0)
            et_archived = latest_gen.get("archived_count", 0)
            
            if et_elites != actual_elites_count:
                results["errors"].append(
                    f"Elites count mismatch: EvolutionTracker={et_elites}, elites.json={actual_elites_count}"
                )
                results["valid"] = False
            
            if et_reserves != actual_reserves_count:
                results["errors"].append(
                    f"Reserves count mismatch: EvolutionTracker={et_reserves}, reserves.json={actual_reserves_count}"
                )
                results["valid"] = False
            
            if et_total != actual_total:
                results["errors"].append(
                    f"Total population mismatch: EvolutionTracker={et_total}, files={actual_total}"
                )
                results["valid"] = False
            
            # Archived count is cumulative, so we can't directly compare
            # But we can check if it's reasonable
            if et_archived < actual_archive_count:
                results["warnings"].append(
                    f"EvolutionTracker archived_count={et_archived} < archive.json={actual_archive_count} "
                    "(archive.json is cumulative, may include more)"
                )
    
    return results


def validate_speciation_metrics(outputs_path: Path, logger) -> Dict[str, Any]:
    """Validate speciation metrics in EvolutionTracker."""
    results = {"valid": True, "errors": [], "warnings": []}
    
    tracker = load_json_file(outputs_path / "EvolutionTracker.json")
    if not tracker or not isinstance(tracker, dict):
        results["errors"].append("Cannot validate speciation metrics: EvolutionTracker.json invalid")
        results["valid"] = False
        return results
    
    generations = tracker.get("generations", [])
    if not generations:
        results["warnings"].append("No generations found in EvolutionTracker")
        return results
    
    for gen in generations:
        gen_num = gen.get("generation_number", -1)
        speciation = gen.get("speciation", {})
        
        if not speciation:
            if gen_num > 0:  # Generation 0 might not have speciation
                results["warnings"].append(f"Generation {gen_num} missing speciation block")
            continue
        
        # Check required fields
        required_fields = [
            "species_count", "active_species_count", "frozen_species_count",
            "reserves_size", "speciation_events", "merge_events", "extinction_events",
            "archived_count", "genomes_updated", "total_population"
        ]
        
        for field in required_fields:
            if field not in speciation:
                results["errors"].append(f"Generation {gen_num}: speciation block missing '{field}'")
                results["valid"] = False
        
        # Validate counts
        species_count = speciation.get("species_count", 0)
        active_count = speciation.get("active_species_count", 0)
        frozen_count = speciation.get("frozen_species_count", 0)
        
        if species_count != active_count + frozen_count:
            results["errors"].append(
                f"Generation {gen_num}: species_count={species_count} != "
                f"active_species_count={active_count} + frozen_species_count={frozen_count}"
            )
            results["valid"] = False
        
        # Check diversity metrics are present (may be 0.0)
        if "inter_species_diversity" not in speciation:
            results["warnings"].append(f"Generation {gen_num}: missing inter_species_diversity")
        if "intra_species_diversity" not in speciation:
            results["warnings"].append(f"Generation {gen_num}: missing intra_species_diversity")
    
    return results


def validate_capacity_enforcement(outputs_path: Path, logger) -> Dict[str, Any]:
    """Validate that species capacity is enforced correctly."""
    results = {"valid": True, "errors": [], "warnings": []}
    
    state = load_json_file(outputs_path / "speciation_state.json")
    elites = load_json_file(outputs_path / "elites.json") or []
    tracker = load_json_file(outputs_path / "genome_tracker.json")
    
    if not state or not isinstance(state, dict):
        results["errors"].append("Cannot validate capacity: speciation_state.json invalid")
        results["valid"] = False
        return results
    
    if not isinstance(elites, list):
        elites = []
    
    # Get species capacity from config
    config = state.get("config", {})
    species_capacity = config.get("species_capacity")
    
    if species_capacity is None:
        results["warnings"].append("species_capacity not found in config")
        return results
    
    # Count genomes per species in elites.json
    species_counts = Counter()
    for g in elites:
        sid = g.get("species_id")
        if sid and sid > 0:
            species_counts[sid] += 1
    
    # Check for species exceeding capacity
    exceeding = []
    for sid, count in species_counts.items():
        if count > species_capacity:
            exceeding.append((sid, count))
    
    if exceeding:
        results["valid"] = False
        for sid, count in sorted(exceeding, key=lambda x: x[1], reverse=True):
            results["errors"].append(
                f"Species {sid}: {count} genomes exceeds capacity {species_capacity} by {count - species_capacity}"
            )
    
    # Also check genome tracker
    if tracker and isinstance(tracker, dict) and "genomes" in tracker:
        tracker_counts = Counter()
        for gid, data in tracker.get("genomes", {}).items():
            sid = data.get("species_id")
            if sid and sid > 0:
                tracker_counts[sid] += 1
        
        tracker_exceeding = []
        for sid, count in tracker_counts.items():
            if count > species_capacity:
                tracker_exceeding.append((sid, count))
        
        if tracker_exceeding:
            results["valid"] = False
            for sid, count in sorted(tracker_exceeding, key=lambda x: x[1], reverse=True):
                results["errors"].append(
                    f"Genome tracker: Species {sid} has {count} genomes (exceeds capacity {species_capacity})"
                )
    
    return results


def validate_cross_file_consistency(outputs_path: Path, logger) -> Dict[str, Any]:
    """Validate consistency between files."""
    results = {"valid": True, "errors": [], "warnings": []}
    
    # Load files
    elites = load_json_file(outputs_path / "elites.json") or []
    reserves = load_json_file(outputs_path / "reserves.json") or []
    archive = load_json_file(outputs_path / "archive.json") or []
    state = load_json_file(outputs_path / "speciation_state.json")
    tracker = load_json_file(outputs_path / "genome_tracker.json")
    
    if not isinstance(elites, list):
        elites = []
    if not isinstance(reserves, list):
        reserves = []
    if not isinstance(archive, list):
        archive = []
    
    # Check for duplicate genome IDs
    all_ids = []
    all_ids.extend([g.get("id") for g in elites if g.get("id")])
    all_ids.extend([g.get("id") for g in reserves if g.get("id")])
    all_ids.extend([g.get("id") for g in archive if g.get("id")])
    
    id_counts = Counter(all_ids)
    duplicates = [gid for gid, count in id_counts.items() if count > 1]
    if duplicates:
        results["errors"].append(f"Duplicate genome IDs found: {duplicates[:10]}")
        results["valid"] = False
    
        # Check genome_tracker vs files
    if tracker and isinstance(tracker, dict) and "genomes" in tracker:
        tracker_genomes = tracker.get("genomes", {})
        
        # Check that all genomes in files are in tracker
        # Convert all IDs to strings for comparison
        file_ids = set(str(gid) for gid in all_ids if gid is not None)
        tracker_ids = set(str(gid) for gid in tracker_genomes.keys())
        
        missing_in_tracker = file_ids - tracker_ids
        if missing_in_tracker:
            # Sample a few IDs to show
            sample_ids = list(missing_in_tracker)[:5]
            results["errors"].append(
                f"Genomes in files but not in tracker: {len(missing_in_tracker)} genomes "
                f"(sample IDs: {sample_ids})"
            )
            results["valid"] = False
        
        # Check species_id consistency
        for genome_id in file_ids & tracker_ids:
            tracker_sid = tracker_genomes[genome_id].get("species_id")
            
            # Find genome in files
            file_sid = None
            for g in elites + reserves + archive:
                if str(g.get("id")) == str(genome_id):
                    file_sid = g.get("species_id")
                    break
            
            if file_sid is not None and tracker_sid is not None and file_sid != tracker_sid:
                results["errors"].append(
                    f"Genome {genome_id}: species_id mismatch - file={file_sid}, tracker={tracker_sid}"
                )
                results["valid"] = False
    
    # Check speciation_state vs elites.json
    if state and isinstance(state, dict):
        species_dict = state.get("species", {})
        for sid_str, sp_data in species_dict.items():
            try:
                sid = int(sid_str)
                expected_size = len([g for g in elites if g.get("species_id") == sid])
                actual_size = sp_data.get("size", 0)
                
                if expected_size != actual_size:
                    results["errors"].append(
                        f"Species {sid}: size mismatch - state={actual_size}, elites.json={expected_size}"
                    )
                    results["valid"] = False
            except (ValueError, TypeError):
                continue
    
    return results


def main():
    """Run comprehensive validation."""
    if len(sys.argv) < 2:
        print("Usage: python validate_outputs_comprehensive.py <outputs_path>")
        sys.exit(1)
    
    outputs_path = Path(sys.argv[1])
    if not outputs_path.exists():
        print(f"Error: Path does not exist: {outputs_path}")
        sys.exit(1)
    
    logger = get_logger("ComprehensiveValidation")
    
    print("=" * 80)
    print(f"COMPREHENSIVE VALIDATION: {outputs_path.name}")
    print("=" * 80)
    print()
    
    all_results = {}
    
    # 1. File structure validation
    print("1. Validating file structure...")
    results = validate_file_structure(outputs_path, logger)
    all_results["file_structure"] = results
    if results["valid"]:
        print("   ✓ File structure validation passed")
    else:
        print(f"   ✗ File structure validation failed: {len(results['errors'])} errors")
        for error in results["errors"][:5]:
            print(f"      - {error}")
    print()
    
    # 2. EvolutionTracker validation
    print("2. Validating EvolutionTracker.json...")
    tracker_path = outputs_path / "EvolutionTracker.json"
    if tracker_path.exists():
        if HAS_ET_VALIDATOR:
            et_results = validate_evolution_tracker_comprehensive(tracker_path, logger)
            all_results["evolution_tracker"] = et_results
            
            all_passed = all(r["valid"] for r in et_results.values())
            if all_passed:
                print("   ✓ EvolutionTracker validation passed")
            else:
                print(f"   ✗ EvolutionTracker validation failed")
                for check_name, result in et_results.items():
                    if not result["valid"]:
                        print(f"      - {check_name}: {len(result['errors'])} errors")
        else:
            print("   ⚠ EvolutionTracker validator not available (import failed)")
            all_results["evolution_tracker"] = {"valid": True, "errors": [], "warnings": ["Validator not available"]}
    else:
        print("   ✗ EvolutionTracker.json not found")
        all_results["evolution_tracker"] = {"error": "File not found"}
    print()
    
    # 3. Genome tracker validation
    print("3. Validating genome_tracker.json...")
    results = validate_genome_tracker(outputs_path, logger)
    all_results["genome_tracker"] = results
    if results["valid"]:
        print("   ✓ Genome tracker validation passed")
    else:
        print(f"   ✗ Genome tracker validation failed: {len(results['errors'])} errors")
        for error in results["errors"][:5]:
            print(f"      - {error}")
    print()
    
    # 4. Distribution rules validation
    print("4. Validating distribution rules...")
    results = validate_distribution_rules(outputs_path, logger)
    all_results["distribution_rules"] = results
    if results["valid"]:
        print("   ✓ Distribution rules validation passed")
    else:
        print(f"   ✗ Distribution rules validation failed: {len(results['errors'])} errors")
        for error in results["errors"][:5]:
            print(f"      - {error}")
    print()
    
    # 5. Population counts validation
    print("5. Validating population counts...")
    results = validate_population_counts(outputs_path, logger)
    all_results["population_counts"] = results
    if results["valid"]:
        print("   ✓ Population counts validation passed")
    else:
        print(f"   ✗ Population counts validation failed: {len(results['errors'])} errors")
        for error in results["errors"][:5]:
            print(f"      - {error}")
    if results["warnings"]:
        for warning in results["warnings"][:3]:
            print(f"      ⚠ {warning}")
    print()
    
    # 6. Speciation metrics validation
    print("6. Validating speciation metrics...")
    results = validate_speciation_metrics(outputs_path, logger)
    all_results["speciation_metrics"] = results
    if results["valid"]:
        print("   ✓ Speciation metrics validation passed")
    else:
        print(f"   ✗ Speciation metrics validation failed: {len(results['errors'])} errors")
        for error in results["errors"][:5]:
            print(f"      - {error}")
    if results["warnings"]:
        for warning in results["warnings"][:3]:
            print(f"      ⚠ {warning}")
    print()
    
    # 7. Capacity enforcement validation
    print("7. Validating capacity enforcement...")
    results = validate_capacity_enforcement(outputs_path, logger)
    all_results["capacity_enforcement"] = results
    if results["valid"]:
        print("   ✓ Capacity enforcement validation passed")
    else:
        print(f"   ✗ Capacity enforcement validation failed: {len(results['errors'])} errors")
        for error in results["errors"][:10]:
            print(f"      - {error}")
    if results["warnings"]:
        for warning in results["warnings"][:3]:
            print(f"      ⚠ {warning}")
    print()
    
    # 8. Cross-file consistency validation
    print("8. Validating cross-file consistency...")
    results = validate_cross_file_consistency(outputs_path, logger)
    all_results["cross_file_consistency"] = results
    if results["valid"]:
        print("   ✓ Cross-file consistency validation passed")
    else:
        print(f"   ✗ Cross-file consistency validation failed: {len(results['errors'])} errors")
        for error in results["errors"][:5]:
            print(f"      - {error}")
    if results["warnings"]:
        for warning in results["warnings"][:3]:
            print(f"      ⚠ {warning}")
    print()
    
    # 9. Speciation consistency validation
    print("9. Validating speciation consistency...")
    if HAS_SPE_VALIDATOR:
        tracker = load_json_file(outputs_path / "EvolutionTracker.json")
        if tracker and isinstance(tracker, dict):
            generations = tracker.get("generations", [])
            if generations:
                latest_gen = max(generations, key=lambda g: g.get("generation_number", -1))
                gen_num = latest_gen.get("generation_number", 0)
                
                is_valid, errors = validate_speciation_consistency(
                    outputs_path, gen_num, logger=logger, expect_temp_empty=True
                )
                all_results["speciation_consistency"] = {"valid": is_valid, "errors": errors}
                
                if is_valid:
                    print("   ✓ Speciation consistency validation passed")
                else:
                    print(f"   ✗ Speciation consistency validation failed: {len(errors)} errors")
                    for error in errors[:5]:
                        print(f"      - {error}")
            else:
                print("   ⚠ No generations found for speciation consistency validation")
                all_results["speciation_consistency"] = {"valid": True, "errors": [], "warnings": ["No generations"]}
        else:
            print("   ✗ Cannot validate: EvolutionTracker.json invalid")
            all_results["speciation_consistency"] = {"valid": False, "errors": ["EvolutionTracker.json invalid"]}
    else:
        print("   ⚠ Speciation validator not available (import failed)")
        all_results["speciation_consistency"] = {"valid": True, "errors": [], "warnings": ["Validator not available"]}
    print()
    
    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_valid = True
    total_errors = 0
    total_warnings = 0
    
    for check_name, result in all_results.items():
        if isinstance(result, dict):
            if "valid" in result:
                if not result["valid"]:
                    all_valid = False
                    total_errors += len(result.get("errors", []))
                total_warnings += len(result.get("warnings", []))
            elif "error" in result:
                all_valid = False
                total_errors += 1
    
    if all_valid:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print(f"✗ VALIDATION FAILED: {total_errors} total errors")
    
    if total_warnings > 0:
        print(f"⚠ {total_warnings} warnings")
    
    print("=" * 80)
    
    # Detailed results
    print("\nDetailed Results:")
    for check_name, result in all_results.items():
        if isinstance(result, dict) and "valid" in result:
            status = "✓" if result["valid"] else "✗"
            error_count = len(result.get("errors", []))
            warning_count = len(result.get("warnings", []))
            print(f"  {status} {check_name}: {error_count} errors, {warning_count} warnings")
            if error_count > 0 and error_count <= 10:
                for error in result["errors"]:
                    print(f"      - {error}")
            elif error_count > 10:
                for error in result["errors"][:10]:
                    print(f"      - {error}")
                print(f"      ... and {error_count - 10} more errors")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
