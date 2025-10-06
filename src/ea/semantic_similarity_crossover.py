"""
semantic_similarity_crossover.py

This module contains the semantic similarity crossover operator for the evolutionary algorithm.
This crossover operator combines semantically similar sentences from two parent prompts.
"""

import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Any
import traceback

try:
    from ea.VariationOperators import VariationOperator
except Exception:
    from VariationOperators import VariationOperator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class SemanticSimilarityCrossover(VariationOperator):
    """
    Semantic similarity crossover operator for prompt recombination.
    
    This crossover operator performs semantic similarity-based crossover between two parent prompts.
    It matches semantically similar sentences and combines them to create new variants.
    
    Attributes:
        name (str): Operator name "SemanticSimilarityCrossover"
        operator_type (str): "crossover" (multiple parents required)
        description (str): Description of the operator's functionality
        logger: Logger instance for debugging and monitoring
        model: Sentence transformer model for semantic embeddings
        
    Methods:
        apply(parent_texts): Generate crossover variants using semantic similarity
        
    Example:
        >>> operator = SemanticSimilarityCrossover()
        >>> parents = ["Write a story about a brave knight", "Create a tale about a courageous warrior"]
        >>> variants = operator.apply(parents)
        >>> print(variants)
        ['Write a story about a brave knight Create a tale about a courageous warrior']
    """
    
    def __init__(self, log_file=None):
        """
        Initialize the semantic similarity crossover operator.
        
        Args:
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Loads SentenceTransformer model "all-MiniLM-L6-v2" for embeddings.
        """
        super().__init__("SemanticSimilarityCrossover", "crossover", "Combines semantically similar sentences from two parents.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def apply(self, parent_data: List[Any]) -> List[str]:
        """
        Generate crossover variants using semantic sentence similarity.
        
        This method:
        1. Validates input format and parent count
        2. Extracts prompts from parent data (handles both strings and genome dictionaries)
        3. Splits parent texts into sentences
        4. Generates sentence embeddings using SentenceTransformer
        5. Matches sentences based on cosine similarity (>0.5 threshold)
        6. Combines matched sentences into a single variant
        7. Returns list with one semantic crossover result
        
        Args:
            parent_data (List[Any]): List of parent genome dictionaries (required)
                - Each dictionary must contain: 'prompt' field
                - Minimum 2 parents required
                
        Returns:
            List[str]: List containing one semantic crossover variant
            
        Raises:
            Warning: If insufficient parents provided, logs warning and returns single parent
            
        Example:
            >>> operator = SemanticSimilarityCrossover()
            >>> parents = ["Write a story about a brave knight", "Create a tale about a courageous warrior"]
            >>> variants = operator.apply(parents)
            >>> print(variants)
            ['Write a story about a brave knight Create a tale about a courageous warrior']
        """
        try:
            # Validate inputs - require genome dictionaries
            if not isinstance(parent_data, list) or len(parent_data) < 2:
                self.logger.error(f"{self.name}: Insufficient parents for crossover. Required: 2, Got: {len(parent_data) if isinstance(parent_data, list) else 'not a list'}")
                return []
            
            parent1_data = parent_data[0]
            parent2_data = parent_data[1]
            
            # Validate that inputs are genome dictionaries
            if not isinstance(parent1_data, dict) or not isinstance(parent2_data, dict):
                self.logger.error(f"{self.name}: Parents must be genome dictionaries with 'prompt' field")
                return []
            
            # Validate required fields
            if "prompt" not in parent1_data or "prompt" not in parent2_data:
                self.logger.error(f"{self.name}: Parents must contain 'prompt' field")
                return []
            
            # Extract prompts from validated genome dictionaries
            parent_texts = [parent1_data.get("prompt", ""), parent2_data.get("prompt", "")]
            self.logger.debug(f"{self.name}: Using genome data format")

            p1_sentences = parent_texts[0].split(". ")
            p2_sentences = parent_texts[1].split(". ")
            
            # Generate embeddings for both sets of sentences
            p1_embeddings = self.model.encode(p1_sentences, convert_to_tensor=True)
            p2_embeddings = self.model.encode(p2_sentences, convert_to_tensor=True)

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

