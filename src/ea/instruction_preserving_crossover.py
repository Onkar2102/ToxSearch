"""
instruction_preserving_crossover.py

LLM-based instruction preserving crossover operator for prompt recombination.
Uses local LLM with task templates and north star metric optimization.

"""

import os
from typing import List, Optional, Dict, Any
import traceback

from .VariationOperators import VariationOperator

from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()


class InstructionPreservingCrossover(VariationOperator):
    """LLM-based instruction preserving crossover operator for prompt recombination."""
    
    def __init__(self, north_star_metric: str, log_file: Optional[str] = None, generator=None):
        """Initialize the instruction preserving crossover operator."""
        super().__init__("InstructionPreservingCrossover", "crossover", 
                        f"LLM-based instruction preserving crossover optimized for {north_star_metric}")
        self.logger = get_logger(self.name, log_file)
        self.north_star_metric = north_star_metric
        
        if generator is not None:
            self.generator = generator
        else:
            from .EvolutionEngine import get_generator
            self.generator = get_generator()

    def _extract_score_from_genome(self, genome: Dict[str, Any]) -> float:
        """Extract the north star metric score from a genome dictionary."""
        from utils import get_system_utils
        _, _, _, _, _extract_north_star_score, _ = get_system_utils()
        return _extract_north_star_score(genome, self.north_star_metric)
    
    def _can_extract_score(self, genome: Dict[str, Any]) -> bool:
        """Check if we can extract a score from the genome."""
        from utils import get_system_utils
        _, _, _, _, _extract_north_star_score, _ = get_system_utils()
        return _extract_north_star_score(genome, self.north_star_metric) > 0

    def _create_crossover_prompt(self, parent1: str, parent2: str, parent1_score: float, parent2_score: float, 
                                parent1_output: str = "", parent2_output: str = "") -> str:
        """
        Create a structured prompt using the task template from config.
        
        Args:
            parent1: First parent prompt text
            parent2: Second parent prompt text  
            parent1_score: North star metric score for parent1
            parent2_score: North star metric score for parent2
            parent1_output: Generated output from parent1 (optional)
            parent2_output: Generated output from parent2 (optional)
            
        Returns:
            Formatted prompt using task template
        """
        
        # Use the task template from config
        try:
            template = self.generator.task_templates.get("crossover", "")
            if template:
                prompt = template.format(
                    north_star_metric=self.north_star_metric,
                    parent1=parent1,
                    parent2=parent2,
                    parent1_score=parent1_score,
                    parent2_score=parent2_score,
                    parent1_output=parent1_output,
                    parent2_output=parent2_output
                )
                return prompt
        except Exception as e:
            self.logger.warning(f"{self.name}: Failed to use task template: {e}")
        

    def _parse_crossover_response(self, response: str) -> str:
        """Parse LLM response to extract crossover variants."""
        try:
            # Extract crossover variant from structured tags
            import re
            variant_match = re.search(r'<variant>(.*?)</variant>', response, re.DOTALL)
            if variant_match:
                variant = variant_match.group(1).strip()
                if variant and len(variant) > 10:
                    return variant
            
            # Fallback: Extract question from response
            return self._extract_question_from_response(response)
        except Exception as e:
            self.logger.debug(f"{self.name}: Failed to parse crossover response: {e}")
            return self._extract_question_from_response(response)
    
    def _extract_question_from_response(self, response: str) -> str:
        """Extract a question from LLM response as fallback parsing."""
        try:
            import re
            # Look for sentences ending with question marks
            questions = re.findall(r'[^.!?]*\?', response)
            if questions:
                for question in questions:
                    question = question.strip()
                    if len(question) > 10 and question.endswith('?'):
                        return question
            
            # Look for sentences that start with question words
            question_patterns = [
                r'(?:How|What|Why|When|Where|Who|Which|Can|Could|Should|Would|Do|Does|Did|Is|Are|Was|Were|Will|Shall)\s+[^.!?]*[.!?]',
                r'(?:How|What|Why|When|Where|Who|Which|Can|Could|Should|Would|Do|Does|Did|Is|Are|Was|Were|Will|Shall)\s+[^.!?]*\?'
            ]
            
            for pattern in question_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    for match in matches:
                        match = match.strip()
                        if len(match) > 10:
                            return match
            
            # Look for any sentence that could be a question
            sentences = re.split(r'[.!?]+', response)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and any(word in sentence.lower() for word in ['how', 'what', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'should', 'would']):
                    return sentence + '?'
            
            return ""
            
        except Exception as e:
            self.logger.debug(f"{self.name}: Failed to extract question from response: {e}")
            return ""

    def apply(self, operator_input: Dict[str, Any]) -> List[str]:
        """
        Generate crossover variants using local LLM with north star metric optimization.
        
        This method:
        1. Validates input format and parent count
        2. Extracts prompts, generated outputs, and scores from parent data
        3. Creates structured prompt with parent outputs and scores for optimization
        4. Uses local LLM to create instruction-preserving crossover variants
        5. Returns a single variant optimized for the north star metric
        
        Args:
            operator_input (Dict[str, Any]): Operator input containing:
                - 'parent_data': List of enriched parent genome dictionaries containing:
                    - 'prompt': Original prompt text for crossover
                    - 'generated_text': Generated output from the prompt (optional)
                    - 'scores': Moderation scores dictionary
                    - 'north_star_score': Primary optimization metric score
                - 'max_variants': Maximum number of variants to generate
            
        Returns:
            List[str]: List with crossover variant text (or empty if failed)
            
        Raises:
            Warning: If insufficient parents provided, logs warning and returns empty
            Error: If LLM call fails, logs error and returns original parent
            
        Example:
            >>> operator = InstructionPreservingCrossover("toxicity")
            >>> input_data = {
            ...     "parent_data": [
            ...         {"prompt": "Write a story", "generated_text": "Once upon a time...", "scores": {"toxicity": 0.1}},
            ...         {"prompt": "Create a tale", "generated_text": "In a faraway land...", "scores": {"toxicity": 0.2}}
            ...     ],
            ...     "max_variants": 1
            ... }
            >>> variants = operator.apply(input_data)
        """
        import time
        start_time = time.time()
        
        try:
            
            # Extract parent data and max_variants
            parent_data = operator_input.get("parent_data", [])
            max_variants = operator_input.get("max_variants", 1)
            
            # Validate inputs - require genome dictionaries with required fields
            if not isinstance(parent_data, list) or len(parent_data) < 2:
                self.logger.error(f"{self.name}: Insufficient parents for crossover. Required: 2, Got: {len(parent_data) if isinstance(parent_data, list) else 'not a list'}")
                return []
            
            parent1_data = parent_data[0]
            parent2_data = parent_data[1]
            
            # Validate that inputs are genome dictionaries
            if not isinstance(parent1_data, dict) or not isinstance(parent2_data, dict):
                self.logger.error(f"{self.name}: Parents must be genome dictionaries with required fields")
                return []
            
            # Validate required fields in slimmed parent data structure
            required_fields = ["prompt", "generated_output"]
            for i, parent_data_item in enumerate([parent1_data, parent2_data], 1):
                for field in required_fields:
                    if field not in parent_data_item:
                        self.logger.error(f"{self.name}: Parent {i} missing required field: {field}")
                        return []
                
                # Validate that we can extract scores (either from scores field or moderation_result)
                if not self._can_extract_score(parent_data_item):
                    self.logger.error(f"{self.name}: Parent {i} missing score data (scores field or moderation_result)")
                    return []
            
            if not self.generator:
                self.logger.error(f"{self.name}: No generator available")
                return []
            
            # Extract parent information from validated genome dictionaries
            parent1 = parent1_data.get("prompt", "")
            parent2 = parent2_data.get("prompt", "")
            parent1_output = parent1_data.get("generated_output", "")
            parent2_output = parent2_data.get("generated_output", "")
            
            # Extract scores from genome data
            parent1_score = self._extract_score_from_genome(parent1_data)
            parent2_score = self._extract_score_from_genome(parent2_data)
            
            self.logger.debug(f"{self.name}: Using genome data with outputs and scores")
            
            # Store debug info
            self._last_parent1 = parent1
            self._last_parent2 = parent2
            self._last_parent1_score = parent1_score
            self._last_parent2_score = parent2_score
            self._last_parent1_output = parent1_output
            self._last_parent2_output = parent2_output
            
            # Create structured prompt
            crossover_prompt = self._create_crossover_prompt(parent1, parent2, parent1_score, parent2_score, 
                                                          parent1_output, parent2_output)
            self._last_crossover_prompt = crossover_prompt
            
            self.logger.debug(f"{self.name}: Generating crossover variants for {self.north_star_metric} optimization")
            self.logger.debug(f"{self.name}: Parent 1 (Score: {parent1_score:.3f}): '{parent1[:50]}...'")
            self.logger.debug(f"{self.name}: Parent 2 (Score: {parent2_score:.3f}): '{parent2[:50]}...'")

            try:
                # Generate response using local LLM with unified task parameters
                response = self.generator.generate_prompt(crossover_prompt, "crossover")
                self._last_raw_response = str(response) if response else ""
                
                if response:
                    # Parse response to extract a single variant
                    variant = self._parse_crossover_response(response)
                    if variant:
                        self.logger.info(f"{self.name}: Generated crossover variant")
                        self._last_variants = [variant]
                        return [variant]
                    else:
                        self.logger.error(f"{self.name}: Failed to parse variant from LLM response")
                        return []
                else:
                    self.logger.error(f"{self.name}: Empty LLM response")
                    return []
            except Exception as e:
                self.logger.error(f"{self.name}: LLM call failed: {e}")
                return []

        except Exception as e:
            self.logger.error(f"{self.name}: apply failed with error: {e}\nTrace: {traceback.format_exc()}")
            return []
        finally:
            end_time = time.time()
            operation_time = end_time - start_time
            # Store timing in a global variable that can be accessed by the caller
            if not hasattr(self, '_last_operation_time'):
                self._last_operation_time = {}
            self._last_operation_time['start_time'] = start_time
            self._last_operation_time['end_time'] = end_time
            self._last_operation_time['duration'] = operation_time

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the last crossover operation.
        
        Returns:
            Dictionary containing debug information
        """
        return {
            "parent1": self._last_parent1,
            "parent2": self._last_parent2,
            "parent1_score": self._last_parent1_score,
            "parent2_score": self._last_parent2_score,
            "parent1_output": self._last_parent1_output,
            "parent2_output": self._last_parent2_output,
            "crossover_prompt": self._last_crossover_prompt,
            "raw_response": self._last_raw_response,
            "variants": self._last_variants,
            "north_star_metric": self.north_star_metric
        }