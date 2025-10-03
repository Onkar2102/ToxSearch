"""
llm_paraphrasing_operator.py


This module contains the LLM-based paraphrasing operator for the evolutionary algorithm.
This mutation operator generates paraphrased variants using OpenAI's GPT-4.

Version: 1.0
"""

import os
from typing import List, Optional, Dict, Any, Tuple
from openai import OpenAI

try:
    from ea.VariationOperators import VariationOperator
except Exception:
    from VariationOperators import VariationOperator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class LLMBasedParaphrasingOperator(VariationOperator):
    """
    Paraphrasing operator using OpenAI's LLM.

    This operator generates multiple paraphrased versions of the input text by leveraging
    OpenAI's language model. The paraphrasing process is guided by a specified optimization
    metric (north_star_metric) to ensure the generated variants align with desired objectives.

    Attributes:
        north_star_metric (str): The optimization metric guiding the paraphrasing process.
        logger: Logger instance for debugging and monitoring.
        client: OpenAI client for interacting with the language model.

    Methods:
        apply(text): Generates paraphrased variants of the input text.

    Example:
        >>> operator = LLMBasedParaphrasingOperator(north_star_metric="engagement")
        >>> variants = operator.apply("Write a story about a brave knight")
        >>> print(variants)
        ['Write a tale about a courageous warrior', 'Compose a narrative about a valiant hero']
    """

    def __init__(self, north_star_metric, log_file=None):
        """
        Initialize the LLM-based paraphrasing operator.
        
        Args:
            north_star_metric (str): The optimization metric for paraphrasing direction
            log_file (str, optional): Path to log file for debugging. Defaults to None.
            
        Note:
            Requires OPENAI_API_KEY environment variable to be set.
        """
        super().__init__("LLMBasedParaphrasing", "mutation", "Uses OpenAI LLM to paraphrase input multiple times with optimization intent.")
        self.north_star_metric = north_star_metric
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name} with north_star_metric: {self.north_star_metric}")
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=10.0  # Add timeout to prevent connection hangs
        )  # Ensure your API key is set in the environment

    def apply(self, text: str) -> List[str]:
        """
        Generate paraphrased variants using OpenAI's LLM.
        
        This method:
        1. Creates paraphrasing instructions based on north_star_metric
        2. Makes multiple API calls to OpenAI GPT-4 with temperature variation
        3. Collects unique paraphrases that differ from original text
        4. Returns up to 3 variants (limited by limit_variants function)
        
        Args:
            text (str): Input text to paraphrase
            
        Returns:
            List[str]: List of paraphrased texts (maximum 3)
            
        Raises:
            Warning: If OpenAI API calls fail, logs warning and continues
            
        Example:
            >>> operator = LLMBasedParaphrasingOperator("engagement")
            >>> variants = operator.apply("Write a story about a brave knight")
            >>> print(variants)
            ['Write a tale about a courageous warrior', 'Compose a narrative about a valiant hero']
        """
        variants = set()
        instruction = f"Paraphrase the following in a way that increases the probability of getting response generated with highest value of {self.north_star_metric} (our north star metric) as much as possible:\n{text}"

        for _ in range(4):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates engaging content"},
                        {"role": "user", "content": instruction}
                    ],
                    temperature=0.9,
                    max_tokens=4096,
                    timeout=10.0  # Add timeout to prevent hanging
                )
                paraphrase = response.choices[0].message.content.strip()
                if paraphrase and paraphrase.lower() != text.lower():
                    variants.add(paraphrase)
                    self.logger.debug(f"{self.name}: Generated variant '{paraphrase}'")
            except Exception as e:
                self.logger.error(f"{self.name}: Failed to generate variant: {e}")
                continue  # Continue to next iteration instead of stopping

        result_variants = list(variants) if variants else [text]
        
        # Limit variants to maximum of 3
        from .operator_helpers import limit_variants
        limited_variants = limit_variants(result_variants, max_variants=3)
        self.logger.debug(f"{self.name}: Total {len(result_variants)} paraphrases generated, limited to {len(limited_variants)} via OpenAI for input: '{text[:60]}...'")
        return limited_variants
