"""
Operator Statistics Tracking

Tracks per-operator duplicates and question mark rejections
for comprehensive monitoring of the evolution process.
"""

from typing import Dict, Any


class OperatorStatistics:
    """
    Tracks per-operator statistics for duplicates and question mark rejections.
    
    Attributes:
        operator_stats: Dictionary mapping operator names to their statistics
    """
    
    def __init__(self):
        self.operator_stats = {}  # {operator_name: {"duplicates_removed": int, "question_mark_rejections": int}}
    
    def _ensure_operator_exists(self, operator_name: str):
        """Ensure operator exists in stats dictionary."""
        if operator_name not in self.operator_stats:
            self.operator_stats[operator_name] = {
                "duplicates_removed": 0,
                "question_mark_rejections": 0
            }
    
    def record_duplicate(self, operator_name: str):
        """Record a duplicate variant removal for the specified operator."""
        self._ensure_operator_exists(operator_name)
        self.operator_stats[operator_name]["duplicates_removed"] += 1
    
    def record_question_mark_rejection(self, operator_name: str):
        """Record a question mark rejection for the specified operator."""
        self._ensure_operator_exists(operator_name)
        self.operator_stats[operator_name]["question_mark_rejections"] += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary format for JSON serialization."""
        return self.operator_stats
    
    def reset(self):
        """Reset all statistics counters."""
        self.operator_stats = {}
