"""
semantic_similarity_crossover.py

Semantic similarity-based crossover operator for prompt recombination.

This module implements a crossover operator that combines two parent prompts
by analyzing semantic similarity between their sentences and creating hybrid
variants that preserve meaningful content from both parents.
"""

from typing import List, Any, Dict
import traceback

from .variation_operators import VariationOperator

from utils import get_custom_logging
from utils.device_utils import get_optimal_device, move_to_device
get_logger, _, _, _ = get_custom_logging()


class SemanticSimilarityCrossover(VariationOperator):
    """Semantic similarity crossover operator for prompt recombination."""
    
    def __init__(self, log_file=None):
        """Initialize the semantic similarity crossover operator."""
        super().__init__("SemanticSimilarityCrossover", "crossover", "Combines semantically similar sentences from two parents.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
        
        # Initialize device detection
        self.device = get_optimal_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Load model with device compatibility
        self.model = self._load_model_with_device_support()
    
    def _load_model_with_device_support(self):
        """Load sentence transformer model with device compatibility"""
        try:
            # Lazy import to avoid torch dependency issues
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Move model to appropriate device using centralized utilities
            model = move_to_device(model, self.device)
            self.logger.info(f"Sentence transformer model using {self.device}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer model: {e}")
            raise RuntimeError(f"Unable to load sentence transformer model: {e}")

    def apply(self, operator_input: Dict[str, Any]) -> List[str]:
        """
        Generate crossover variants using semantic sentence similarity.
        
        This method:
        1. Validates input format and parent count
        2. Extracts parent data from operator input
        3. Splits parent texts into sentences
        4. Generates sentence embeddings using SentenceTransformer
        5. Matches sentences based on cosine similarity (>0.5 threshold)
        6. Combines matched sentences into a single variant
        7. Returns list with semantic crossover results
        
        Args:
            operator_input (Dict[str, Any]): Operator input containing:
                - 'parent_data': List of enriched parent genome dictionaries containing:
                    - 'prompt': Original prompt text for crossover
                    - 'generated_text': Generated output from the prompt (optional)
                    - 'scores': Moderation scores dictionary
                    - 'north_star_score': Primary optimization metric score
                - 'max_variants': Maximum number of variants to generate
                
        Returns:
            List[str]: List containing semantic crossover variants
            
        Raises:
            Warning: If insufficient parents provided, logs warning and returns single parent
            
        Example:
            >>> operator = SemanticSimilarityCrossover()
            >>> input_data = {
            ...     "parent_data": [
            ...         {"prompt": "Write a story about a brave knight"},
            ...         {"prompt": "Create a tale about a courageous warrior"}
            ...     ],
            ...     "max_variants": 2
            ... }
            >>> variants = operator.apply(input_data)
            >>> print(variants)
            ['Write a tale about a brave knight']
        """
        try:
            # Lazy imports to avoid torch dependency issues
            import torch
            from sentence_transformers import util
            
            # Validate inputs - require operator input dictionary
            if not isinstance(operator_input, dict):
                self.logger.error(f"{self.name}: Input must be a dictionary")
                return []
            
            # Extract parent data and max_variants
            parent_data = operator_input.get("parent_data", [])
            max_variants = operator_input.get("max_variants", 1)
            
            if not isinstance(parent_data, list) or len(parent_data) < 2:
                self.logger.error(f"{self.name}: Insufficient parents for crossover. Required: 2, Got: {len(parent_data) if isinstance(parent_data, list) else 'not a list'}")
                return []
            
            parent1_data = parent_data[0]
            parent2_data = parent_data[1]
            
            # Validate that inputs are enriched parent dictionaries
            if not isinstance(parent1_data, dict) or not isinstance(parent2_data, dict):
                self.logger.error(f"{self.name}: Parents must be enriched parent dictionaries")
                return []
            
            # Validate required fields
            if "prompt" not in parent1_data or "prompt" not in parent2_data:
                self.logger.error(f"{self.name}: Parents must contain 'prompt' field")
                return []
            
            # Extract prompts from validated parent dictionaries
            parent_texts = [parent1_data.get("prompt", ""), parent2_data.get("prompt", "")]
            self.logger.debug(f"{self.name}: Using enriched parent data format")

            p1_sentences = parent_texts[0].split(". ")
            p2_sentences = parent_texts[1].split(". ")
            
            # Generate embeddings for both sets of sentences with device compatibility
            try:
                p1_embeddings = self.model.encode(p1_sentences, convert_to_tensor=True, device=self.device)
                p2_embeddings = self.model.encode(p2_sentences, convert_to_tensor=True, device=self.device)
            except Exception as e:
                self.logger.warning(f"GPU embedding generation failed, falling back to CPU: {e}")
                # Fallback to CPU if GPU embedding fails
                p1_embeddings = self.model.encode(p1_sentences, convert_to_tensor=True, device="cpu")
                p2_embeddings = self.model.encode(p2_sentences, convert_to_tensor=True, device="cpu")

            matched_sentences = []
            for i, emb1 in enumerate(p1_embeddings):
                similarities = util.cos_sim(emb1, p2_embeddings)[0]
                # Use torch.argmax to avoid numpy/tensor conversion issues on GPU
                top_idx = int(torch.argmax(similarities).item())
                sim_score = similarities[top_idx].item()
                if sim_score > 0.5:
                    matched_sentences.append(p1_sentences[i])
                    matched_sentences.append(p2_sentences[top_idx])

            # Clean up sentences and concatenate properly
            cleaned_sentences = []
            for sentence in matched_sentences:
                sentence = sentence.strip()
                if sentence:
                    # Ensure sentence ends with proper punctuation
                    if not sentence.endswith(('.', '!', '?')):
                        sentence += '.'
                    cleaned_sentences.append(sentence)
            
            # Join sentences with proper spacing
            result = " ".join(cleaned_sentences).strip()
            
            # Final cleanup - ensure proper spacing after periods
            result = result.replace('. ', '. ').replace('..', '.').strip()

            self.logger.debug(f"{self.name}: Created crossover from {len(cleaned_sentences)} semantically matched sentences.")
            return [result] if result and result != "." else [parent_texts[0]]
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed with error: {e}\nTrace: {traceback.format_exc()}")
            return parent_texts[:1] if parent_texts else []

