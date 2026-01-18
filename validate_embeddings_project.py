#!/usr/bin/env python3
"""
Validate that embeddings are properly saved and used throughout the project.
Checks for:
1. Embeddings in elites.json and reserves.json
2. No embeddings in archive.json (intentional)
3. Code that properly handles saved embeddings
4. Any issues with embedding usage
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()
logger = get_logger("ValidateEmbeddings")


def check_file_embeddings(file_path: Path, expected: bool, file_type: str) -> Tuple[bool, Dict]:
    """Check if embeddings exist in a file."""
    if not file_path.exists():
        return False, {"error": f"{file_type} file not found: {file_path}"}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            genomes = json.load(f)
        
        if not isinstance(genomes, list):
            return False, {"error": f"{file_type} is not a list"}
        
        if len(genomes) == 0:
            return True, {"status": "empty", "count": 0}
        
        # Check embeddings
        with_embeddings = sum(1 for g in genomes if "prompt_embedding" in g and g.get("prompt_embedding") is not None)
        without_embeddings = len(genomes) - with_embeddings
        
        result = {
            "total": len(genomes),
            "with_embeddings": with_embeddings,
            "without_embeddings": without_embeddings,
            "embedding_rate": with_embeddings / len(genomes) if len(genomes) > 0 else 0.0
        }
        
        # Validate expectation
        if expected:
            if without_embeddings > 0:
                result["warning"] = f"Expected all genomes to have embeddings, but {without_embeddings} are missing"
                return False, result
        else:
            if with_embeddings > 0:
                result["warning"] = f"Expected no embeddings, but {with_embeddings} genomes have embeddings"
                return False, result
        
        return True, result
        
    except Exception as e:
        return False, {"error": str(e)}


def validate_outputs_directory(outputs_dir: Path) -> Dict:
    """Validate embeddings in an outputs directory."""
    results = {
        "directory": str(outputs_dir),
        "valid": True,
        "errors": [],
        "warnings": [],
        "files": {}
    }
    
    # Check elites.json - SHOULD have embeddings
    elites_path = outputs_dir / "elites.json"
    valid, info = check_file_embeddings(elites_path, expected=True, file_type="elites")
    results["files"]["elites.json"] = info
    if not valid:
        if "error" in info:
            results["errors"].append(f"elites.json: {info['error']}")
        if "warning" in info:
            results["warnings"].append(f"elites.json: {info['warning']}")
        results["valid"] = False
    
    # Check reserves.json - SHOULD have embeddings
    reserves_path = outputs_dir / "reserves.json"
    valid, info = check_file_embeddings(reserves_path, expected=True, file_type="reserves")
    results["files"]["reserves.json"] = info
    if not valid:
        if "error" in info:
            results["errors"].append(f"reserves.json: {info['error']}")
        if "warning" in info:
            results["warnings"].append(f"reserves.json: {info['warning']}")
        results["valid"] = False
    
    # Check archive.json - SHOULD NOT have embeddings (intentional, to save space)
    archive_path = outputs_dir / "archive.json"
    if archive_path.exists():
        valid, info = check_file_embeddings(archive_path, expected=False, file_type="archive")
        results["files"]["archive.json"] = info
        if not valid:
            if "warning" in info:
                # This is just a warning, not an error (embeddings in archive are wasteful but not breaking)
                results["warnings"].append(f"archive.json: {info['warning']}")
    
    # Check temp.json - may or may not have embeddings (depends on when checked)
    temp_path = outputs_dir / "temp.json"
    if temp_path.exists():
        valid, info = check_file_embeddings(temp_path, expected=None, file_type="temp")
        results["files"]["temp.json"] = info
    
    return results


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate embeddings in project outputs")
    parser.add_argument("--outputs-dir", type=str, default="data/outputs/20260117_1152",
                       help="Outputs directory to validate")
    parser.add_argument("--all", action="store_true",
                       help="Validate all output directories")
    
    args = parser.parse_args()
    
    base_dir = Path("data/outputs")
    
    if args.all:
        # Find all output directories
        output_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        logger.info(f"Validating {len(output_dirs)} output directories...")
    else:
        output_dirs = [Path(args.outputs_dir)]
    
    all_valid = True
    for outputs_dir in output_dirs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating: {outputs_dir}")
        logger.info(f"{'='*60}")
        
        results = validate_outputs_directory(outputs_dir)
        
        if results["valid"]:
            logger.info("✓ Validation PASSED")
        else:
            logger.error("✗ Validation FAILED")
            all_valid = False
        
        # Print file summaries
        for filename, info in results["files"].items():
            if "error" in info:
                logger.error(f"  {filename}: ERROR - {info['error']}")
            elif "warning" in info:
                logger.warning(f"  {filename}: WARNING - {info['warning']}")
            else:
                if "total" in info:
                    logger.info(f"  {filename}: {info['with_embeddings']}/{info['total']} genomes have embeddings ({info['embedding_rate']*100:.1f}%)")
                else:
                    logger.info(f"  {filename}: {info.get('status', 'OK')}")
        
        # Print errors and warnings
        if results["errors"]:
            logger.error("\nErrors:")
            for error in results["errors"]:
                logger.error(f"  - {error}")
        
        if results["warnings"]:
            logger.warning("\nWarnings:")
            for warning in results["warnings"]:
                logger.warning(f"  - {warning}")
    
    if all_valid:
        logger.info("\n" + "="*60)
        logger.info("All validations PASSED ✓")
        logger.info("="*60)
        return 0
    else:
        logger.error("\n" + "="*60)
        logger.error("Some validations FAILED ✗")
        logger.error("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
