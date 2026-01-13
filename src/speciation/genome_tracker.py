"""
genome_tracker.py

Genome audit trail tracking for speciation pipeline.
Tracks individual genome movements through the clustering and distribution process.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class GenomeTracker:
    """
    Tracks individual genome movements through the speciation pipeline.
    
    Provides audit trail for:
    - Clustering assignments (species_id assignment)
    - Capacity enforcement (archival events)
    - Species transitions (merges, extinctions)
    - Cluster 0 movements (outlier assignment, speciation from cluster 0)
    
    Events are logged with timestamps and details for post-hoc analysis.
    """
    
    def __init__(self, generation: int, logger=None):
        """
        Initialize genome tracker for a generation.
        
        Args:
            generation: Current generation number
            logger: Optional logger instance
        """
        self.generation = generation
        self.events: List[Dict[str, Any]] = []
        self.logger = logger or get_logger("GenomeTracker")
    
    def log(self, genome_id: str, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a genome event.
        
        Args:
            genome_id: Unique genome identifier
            event: Event type (e.g., "clustering_assigned", "capacity_archived", "species_merged")
            details: Optional event details (e.g., {"species_id": 1, "reason": "capacity"})
        """
        event_record = {
            "genome_id": genome_id,
            "event": event,
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.events.append(event_record)
        self.logger.debug(f"Genome {genome_id}: {event} (gen {self.generation})")
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save genome tracker events to JSON file.
        
        Args:
            path: Optional path to save file. If None, uses default outputs_path / "genome_tracker_gen_{generation}.json"
        """
        from utils import get_system_utils
        _, _, _, get_outputs_path, _, _ = get_system_utils()
        
        if path is None:
            outputs_path = get_outputs_path()
            path = str(outputs_path / f"genome_tracker_gen_{self.generation}.json")
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path_obj, 'w', encoding='utf-8') as f:
                json.dump({
                    "generation": self.generation,
                    "total_events": len(self.events),
                    "events": self.events
                }, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved genome tracker with {len(self.events)} events to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save genome tracker to {path}: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of tracked events.
        
        Returns:
            Dictionary with event counts and statistics
        """
        event_counts = {}
        for event in self.events:
            event_type = event["event"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "generation": self.generation,
            "total_events": len(self.events),
            "unique_genomes": len(set(e["genome_id"] for e in self.events)),
            "event_counts": event_counts
        }
