## @file VariationOperators.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Abstract base class defining the interface for all variation operators in the evolutionary pipeline.

from abc import ABC, abstractmethod
from typing import List
import logging
logger = logging.getLogger(__name__)

## @class VariationOperator
# @brief Abstract base class for variation operators (e.g., mutation, crossover) used in prompt evolution.
class VariationOperator(ABC):

    ## @brief Initializes the operator with a name, type, and description.
    # @param name Name of the operator (defaults to class name).
    # @param operator_type Operator category ('mutation', 'crossover', or 'hybrid').
    # @param description Short description of the operator's functionality.
    # @return None
    def __init__(self, name=None, operator_type="mutation", description=""):
        self.name = name or self.__class__.__name__
        self.operator_type = operator_type
        self.description = description
        logger.debug(f"Initialized operator: {self.name} (type={self.operator_type})")

    ## @brief Applies the variation to the input text.
    # @param text Input prompt string to be mutated.
    # @return List of modified output strings (variants).
    @abstractmethod
    def apply(self, text: str) -> List[str]:
        pass

    ## @brief Returns the operator’s name and type as a string.
    # @return String formatted as "name (type)".
    def __str__(self):
        logger.debug(f"__str__ called on operator: {self.name}")
        return f"{self.name} ({self.operator_type})"

    ## @brief Retrieves metadata about the operator for logging or tracking.
    # @return Dictionary with keys: 'name', 'type', 'description'.
    def get_metadata(self) -> dict:
        logger.debug(f"get_metadata called for operator: {self.name}")
        return {
            "name": self.name,
            "type": self.operator_type,
            "description": self.description
        }