"""
LLM-based POS-aware antonym replacement for text mutation.

"""

from typing import List, Optional, Dict, Any, Tuple
import random
import re
import json
import spacy
from dataclasses import dataclass

from .VariationOperators import VariationOperator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()

# spaCy pipeline loaded once
nlp = spacy.load("en_core_web_sm")


@dataclass
class POSWord:
    """POS-tagged word with position information."""
    word: str
    start: int
    end: int
    pos_tag: str
    pos_description: str


class LLM_POSAwareAntonymReplacement(VariationOperator):
    """
    LLM-based antonym replacement with POS awareness.
    
    Detects POS tags, generates antonyms via LLM, and creates text variants.
    """

    # POS inventory (excluding PUNCT, SYM, X)
    POS_DESCRIPTIONS = {
        "ADJ": "Adjective: noun modifiers describing properties",
        "ADV": "Adverb: verb modifiers of time, place, manner",
        "NOUN": "words for persons, places, things, etc.",
        "VERB": "words for actions and processes",
        "PROPN": "Proper noun: name of a person, organization, place, etc.",
        "INTJ": "Interjection: exclamation, greeting, yes/no response, etc.",
        "ADP": "Adposition (Preposition/Postposition): marks a noun's spatial, temporal, or other relation",
        "AUX": "Auxiliary: helping verb marking tense, aspect, mood, etc.",
        "CCONJ": "Coordinating Conjunction: joins two phrases/clauses",
        "DET": "Determiner: marks noun phrase properties",
        "NUM": "Numeral",
        "PART": "Particle: a function word that must be associated with another word",
        "PRON": "Pronoun: a shorthand for referring to an entity or event",
        "SCONJ": "Subordinating Conjunction: joins a main clause with a subordinate clause such as a sentential complement"
    }

    def __init__(self, log_file: Optional[str] = None, max_variants: int = 3, num_POS_tags: int = 1, seed: Optional[int] = 42, generator=None):
        """
        Initialize the LLM POS-aware antonym replacement operator.
        
        Args:
            log_file: Path to log file (optional)
            max_variants: Maximum number of variants to generate (default: 3)
            num_POS_tags: Number of POS types to randomly select (1 to max available)
            seed: Random seed for reproducible selection (default: 42)
            generator: LLaMA generator instance to use. If None, will create own instance.
        """
        super().__init__(
            "LLM_POSAwareAntonymReplacement", 
            "mutation", 
            "Step 1: POS-aware detection and validation"
        )
        
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized {self.name}")
        
        # Validate and set parameters
        self.max_variants = self._validate_max_variants(max_variants)
        self.num_POS_tags = self._validate_num_POS_tags(num_POS_tags)
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Initialize generator - use provided or create new one
        if generator is not None:
            self.generator = generator
            self.logger.info(f"{self.name}: Using provided LLM generator")
        else:
            from .EvolutionEngine import get_generator
            self.generator = get_generator()
            self.logger.debug(f"{self.name}: LLM generator initialized successfully")
        
        self.logger.debug(f"{self.name}: Configured with max_variants={self.max_variants}, num_POS_tags={self.num_POS_tags}, seed={seed}")

    def _validate_max_variants(self, max_variants: int) -> int:
        """Ensure max_variants is positive integer."""
        try:
            val = max(1, int(max_variants))
            if val < 1:
                self.logger.warning(f"{self.name}: max_variants < 1, setting to 1")
                return 1
            return val
        except (ValueError, TypeError):
            self.logger.warning(f"{self.name}: Invalid max_variants '{max_variants}', using default 1")
            return 1

    def _validate_num_POS_tags(self, num_POS_tags: int) -> int:
        """Ensure num_POS_tags is within valid range."""
        try:
            val = max(1, int(num_POS_tags))
            max_available = len(self.POS_DESCRIPTIONS)
            if val > max_available:
                self.logger.warning(f"{self.name}: num_POS_tags={val} > max_available={max_available}, capping to {max_available}")
                return max_available
            return val
        except (ValueError, TypeError):
            self.logger.warning(f"{self.name}: Invalid num_POS_tags '{num_POS_tags}', using default 1")
            return 1

    def _detect_and_organize_pos(self, text: str) -> Dict[str, List[POSWord]]:
        """
        Detect POS tags and organize by type.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict mapping POS_tag -> List[POSWord objects]
        """
        self.logger.debug(f"{self.name}: Starting POS detection for text: '{text[:50]}...'")
        
        try:
            # Process text with spaCy
            doc = nlp(text)
            
            # Initialize organization structure
            pos_words: Dict[str, List[POSWord]] = {}
            
            # Extract POS information for each token
            for token in doc:
                pos_tag = token.pos_.upper()
                
                # Filter valid POS tags and alphabetic tokens
                if pos_tag in self.POS_DESCRIPTIONS and token.is_alpha:
                    pos_word = POSWord(
                        word=token.text,
                        start=token.idx,
                        end=token.idx + len(token.text),
                        pos_tag=pos_tag,
                        pos_description=self.POS_DESCRIPTIONS[pos_tag]
                    )
                    
                    if pos_tag not in pos_words:
                        pos_words[pos_tag] = []
                    pos_words[pos_tag].append(pos_word)
            
            self.logger.info(f"{self.name}: POS detection complete. Found {len(pos_words)} POS types:")
            for pos_tag, words in pos_words.items():
                self.logger.info(f"{self.name}:   {pos_tag} ({len(words)} instances): {[w.word for w in words[:3]]}{'...' if len(words) > 3 else ''}")
            
            return pos_words
            
        except Exception as e:
            self.logger.error(f"{self.name}: POS detection failed: {e}")
            return {}

    def _select_pos_types(self, detected_pos: Dict[str, List[POSWord]]) -> List[str]:
        """
        Randomly select POS types up to num_POS_tags limit.
        
        Args:
            detected_pos: POS words organized by type
            
        Returns:
            List of selected POS tag strings
        """
        available_pos = list(detected_pos.keys())
        
        if not available_pos:
            self.logger.warning(f"{self.name}: No POS detected, returning empty selection")
            return []
        
        # Select up to num_POS_tags POS types
        num_to_select = min(self.num_POS_tags, len(available_pos))
        selected_pos = self.rng.sample(available_pos, num_to_select)
        
        self.logger.info(f"{self.name}: Selected {len(selected_pos)} POS types: {selected_pos}")
        return selected_pos

    def _create_antonym_prompt(self, pos_tag: str, pos_description: str, sample_words: List[str], context_text: str) -> str:
        """Create a prompt for LLM to generate antonyms using config template."""
        
        sample_words_str = ", ".join(sample_words[:5])
        
        # Get template from config
        template = self.generator.task_templates.get("antonym_generation", "")
        if template:
            prompt = template.format(
                pos_tag=pos_tag,
                pos_description=pos_description,
                sample_words_str=sample_words_str,
                context_text=context_text[:100] + ('...' if len(context_text) > 100 else ''),
                max_variants=self.max_variants
            )
            return prompt
        
        # Fallback to original prompt if template not found
        return f"""You are a linguistic expert. I need antonyms or opposites for words with the same grammatical function.

POS Type: {pos_tag} ({pos_description})
Sample words from the text: {sample_words_str}
Context: "{context_text[:100]}{'...' if len(context_text) > 100 else ''}"

Please provide exactly {self.max_variants} antonyms/opposites that:
1. Have the same POS tag ({pos_tag})
2. Are antonyms or opposites of the sample words OR other words with opposite meaning but same grammatical function

Return ONLY a JSON array of words, like this:
["word1", "word2", "word3"]

Antonyms for {pos_tag}:
"""

    def _parse_antonyms_from_response(self, response: str, pos_tag: str) -> List[str]:
        """Parse antonyms from LLM response."""
        try:
            # Extract antonyms from structured tags
            import re
            antonyms_match = re.search(r'<antonyms>(.*?)</antonyms>', response, re.DOTALL)
            if antonyms_match:
                antonyms_text = antonyms_match.group(1).strip()
                antonyms = json.loads(antonyms_text)
                if isinstance(antonyms, list):
                    cleaned_antonyms = []
                    for word in antonyms:
                        if isinstance(word, str) and word.strip():
                            cleaned_word = word.strip().lower()
                            if len(cleaned_word) > 1 and cleaned_word.isalpha():
                                cleaned_antonyms.append(cleaned_word)
                    return cleaned_antonyms[:self.max_variants]
            
            # Fallback: Extract words from response text
            return self._extract_words_from_response(response)
            
        except Exception as e:
            self.logger.debug(f"{self.name}: Failed to parse antonyms from response: {e}")
            return self._extract_words_from_response(response)
    
    def _extract_words_from_response(self, response: str) -> List[str]:
        """Extract words from LLM response as fallback parsing."""
        try:
            import re
            # Look for JSON arrays in the response
            json_match = re.search(r'\[(.*?)\]', response)
            if json_match:
                json_text = '[' + json_match.group(1) + ']'
                try:
                    words = json.loads(json_text)
                    if isinstance(words, list):
                        cleaned_words = []
                        for word in words:
                            if isinstance(word, str) and word.strip():
                                cleaned_word = word.strip().lower()
                                if len(cleaned_word) > 1 and cleaned_word.isalpha():
                                    cleaned_words.append(cleaned_word)
                        return cleaned_words[:self.max_variants]
                except:
                    pass
            
            # Look for quoted words
            quoted_words = re.findall(r'"([^"]+)"', response)
            if quoted_words:
                cleaned_words = []
                for word in quoted_words:
                    stripped_word = word.strip()
                    if stripped_word and len(stripped_word) > 1 and stripped_word.isalpha():
                        cleaned_words.append(stripped_word.lower())
                return cleaned_words[:self.max_variants]
            
            # Look for comma-separated words
            words = re.findall(r'\b[a-zA-Z]{2,}\b', response)
            if words:
                cleaned_words = []
                for word in words:
                    if len(word) > 1 and word.isalpha():
                        cleaned_words.append(word.lower())
                return cleaned_words[:self.max_variants]
            
            return []
            
        except Exception as e:
            self.logger.debug(f"{self.name}: Failed to extract words from response: {e}")
            return []

    def _ask_llm_for_antonyms(self, pos_tag: str, pos_words: List[POSWord], text_context: str) -> List[str]:
        """
        Generate antonyms for a POS type using LLM.
        
        Args:
            pos_tag: The POS tag (e.g., "ADJ", "VERB")
            pos_words: List of POSWord objects for this POS type
            text_context: The original text for context
            
        Returns:
            List of antonym words with the same POS tag
        """
        if not self.generator:
            self.logger.warning(f"{self.name}: LLM generator unavailable, skipping antonym generation")
            return []
        
        try:
            # Extract unique words for this POS type
            unique_words = list(set(word.word for word in pos_words))
            pos_description = self.POS_DESCRIPTIONS[pos_tag]
            
            # Create prompt for LLM
            prompt = self._create_antonym_prompt(pos_tag, pos_description, unique_words, text_context)
            
            self.logger.debug(f"{self.name}: Asking LLM for {pos_tag} antonyms")
            self.logger.debug(f"{self.name}: Prompt: {prompt[:200]}...")
            
            # Get LLM response with unified task parameters
            response = self.generator.generate_prompt(prompt, "antonym_generation")
            
            if not response:
                self.logger.warning(f"{self.name}: Empty LLM response for {pos_tag}")
                return []
            
            # Parse LLM response
            antonyms_data = self._parse_antonyms_from_response(response, pos_tag)
            
            if antonyms_data:
                self.logger.info(f"{self.name}: Generated antonyms for {pos_tag}: {len(antonyms_data)} words")
                return antonyms_data
            else:
                self.logger.warning(f"{self.name}: Failed to parse antonyms for {pos_tag}")
                return []
                
        except Exception as e:
            self.logger.error(f"{self.name}: LLM antonym generation failed for {pos_tag}: {e}")
            return []

    def _generate_antonyms_for_selected_pos(self, detected_pos: Dict[str, List[POSWord]], selected_pos: List[str], text: str) -> Dict[str, List[str]]:
        """
        Generate antonyms for all selected POS types.
        
        Args:
            detected_pos: POS words organized by type
            selected_pos: List of selected POS types
            text: Original text for context
            
        Returns:
            Dict mapping POS_tag -> List[antonyms]
        """
        antonyms_by_pos = {}
        
        self.logger.info(f"{self.name}: STEP 2 - Generating antonyms for {len(selected_pos)} POS types")
        
        for pos_tag in selected_pos:
            if pos_tag in detected_pos:
                pos_words = detected_pos[pos_tag]
                antonyms = self._ask_llm_for_antonyms(pos_tag, pos_words, text)
                
                if antonyms:
                    antonyms_by_pos[pos_tag] = antonyms
                    self.logger.info(f"{self.name}: Generated {len(antonyms)} antonyms for {pos_tag}: {antonyms[:3]}{'...' if len(antonyms) > 3 else ''}")
                else:
                    self.logger.warning(f"{self.name}: No antonyms generated for {pos_tag}")
            else:
                self.logger.warning(f"{self.name}: POS tag {pos_tag} not found in detected POS")
        
        self.logger.info(f"{self.name}: STEP 2 COMPLETE - Generated antonyms for {len(antonyms_by_pos)} POS types")
        return antonyms_by_pos

    def _generate_text_variants(self, text: str, detected_pos: Dict[str, List[POSWord]], antonyms_by_pos: Dict[str, List[str]]) -> List[str]:
        """
        Generate text variants by substituting antonyms.
        
        Args:
            text: Original text
            detected_pos: POS words organized by type
            antonyms_by_pos: Antonyms generated for each POS type
            
        Returns:
            List of text variants with substitutions
        """
        self.logger.info(f"{self.name}: STEP 3 - Generating text variants")
        
        try:
            variants = []
            for variant_num in range(self.max_variants):
                variant = self._create_single_variant(text, detected_pos, antonyms_by_pos, variant_num)
                if variant and variant != text:
                    variants.append(variant)
                    self.logger.debug(f"{self.name}: Generated variant {len(variants)}: '{variant[:30]}...'")
            
            if len(variants) < self.max_variants:
                additional_variants = self._generate_additional_variants(text, detected_pos, antonyms_by_pos, variants)
                variants.extend(additional_variants)
            
            # Remove duplicates
            unique_variants = []
            seen = set()
            for variant in variants:
                if variant not in seen:
                    unique_variants.append(variant)
                    seen.add(variant)
            
            # Limit to max_variants
            final_variants = unique_variants[:self.max_variants]
            
            self.logger.info(f"{self.name}: STEP 3 COMPLETE - Generated {len(final_variants)} unique variants")
            return final_variants
            
        except Exception as e:
            self.logger.error(f"{self.name}: Step 3 variant generation failed: {e}")
            return []

    def _create_single_variant(self, text: str, detected_pos: Dict[str, List[POSWord]], antonyms_by_pos: Dict[str, List[str]], variant_num: int) -> str:
        """
        Create a single text variant by substituting antonyms.
        
        Args:
            text: Original text
            detected_pos: POS words organized by type
            antonyms_by_pos: Antonyms for each POS type
            variant_num: Variant number (for different substitution strategies)
            
        Returns:
            Single text variant
        """
        try:
            variant_text = text
            
            # Sort POS types for consistent ordering
            pos_types = sorted(antonyms_by_pos.keys())
            
            # Apply substitutions for each POS type
            for pos_tag in pos_types:
                if pos_tag in detected_pos and pos_tag in antonyms_by_pos:
                    pos_words = detected_pos[pos_tag]
                    antonyms = antonyms_by_pos[pos_tag]
                    
                    if antonyms:  # Only proceed if we have antonyms
                        # Select antonym based on variant number
                        antonym_index = variant_num % len(antonyms)
                        selected_antonym = antonyms[antonym_index]
                        
                        # Apply substitution for this POS type
                        variant_text = self._substitute_pos_words(variant_text, pos_words, selected_antonym, pos_tag)
            
            return variant_text
            
        except Exception as e:
            self.logger.error(f"{self.name}: Single variant creation failed: {e}")
            return text

    def _substitute_pos_words(self, text: str, pos_words: List[POSWord], antonym: str, pos_tag: str) -> str:
        """
        Substitute words of a specific POS type with an antonym.
        
        Args:
            text: Current text
            pos_words: List of POSWord objects to potentially replace
            antonym: Antonym to use for replacement
            pos_tag: POS tag for context
            
        Returns:
            Text with substitutions applied
        """
        try:
            if not pos_words or not antonym:
                return text
            
            # Sort words by position (reverse order to avoid position shifts)
            sorted_words = sorted(pos_words, key=lambda w: w.start, reverse=True)
            
            result_text = text
            
            for word_obj in sorted_words:
                # Validate word boundaries
                if self._is_valid_word_boundary(text, word_obj.start, word_obj.end):
                    # Perform substitution
                    result_text = self._safe_substitute(result_text, word_obj.start, word_obj.end, antonym)
                    self.logger.debug(f"{self.name}: Replaced '{word_obj.word}' with '{antonym}' at position {word_obj.start}-{word_obj.end}")
            
            return result_text
            
        except Exception as e:
            self.logger.error(f"{self.name}: POS word substitution failed for {pos_tag}: {e}")
            return text

    def _is_valid_word_boundary(self, text: str, start: int, end: int) -> bool:
        """
        Validate that the word boundaries are correct and safe for substitution.
        
        Args:
            text: Original text
            start: Start position
            end: End position
            
        Returns:
            True if boundaries are valid
        """
        try:
            # Check basic bounds
            if start < 0 or end > len(text) or start >= end:
                return False
            
            # Check that the substring matches expected word
            word = text[start:end]
            if not word.strip():
                return False
            
            # Check for word boundaries (not in middle of another word)
            if start > 0 and text[start-1].isalnum():
                return False
            if end < len(text) and text[end].isalnum():
                return False
            
            return True
            
        except Exception:
            return False

    def _safe_substitute(self, text: str, start: int, end: int, replacement: str) -> str:
        """
        Safely substitute text at given positions.
        
        Args:
            text: Original text
            start: Start position
            end: End position
            replacement: Replacement text
            
        Returns:
            Text with substitution applied
        """
        try:
            # Validate inputs
            if not isinstance(text, str) or not isinstance(replacement, str):
                return text
            
            if start < 0 or end > len(text) or start >= end:
                return text
            
            # Perform substitution
            return text[:start] + replacement + text[end:]
            
        except Exception as e:
            self.logger.error(f"{self.name}: Safe substitution failed: {e}")
            return text

    def _generate_additional_variants(self, text: str, detected_pos: Dict[str, List[POSWord]], antonyms_by_pos: Dict[str, List[str]], existing_variants: List[str]) -> List[str]:
        """
        Generate additional variants using different substitution strategies.
        
        Args:
            text: Original text
            detected_pos: POS words organized by type
            antonyms_by_pos: Antonyms for each POS type
            existing_variants: Already generated variants
            
        Returns:
            List of additional variants
        """
        additional_variants = []
        
        try:
            # Strategy: Substitute only one POS type per variant
            for pos_tag in antonyms_by_pos.keys():
                if pos_tag in detected_pos:
                    pos_words = detected_pos[pos_tag]
                    antonyms = antonyms_by_pos[pos_tag]
                    
                    for antonym in antonyms:
                        # Create variant with only this POS type substituted
                        variant = self._substitute_pos_words(text, pos_words, antonym, pos_tag)
                        
                        if variant != text and variant not in existing_variants and variant not in additional_variants:
                            additional_variants.append(variant)
                            
                        # Stop if we have enough variants
                        if len(existing_variants) + len(additional_variants) >= self.max_variants:
                            break
                    
                    if len(existing_variants) + len(additional_variants) >= self.max_variants:
                        break
            
            return additional_variants
            
        except Exception as e:
            self.logger.error(f"{self.name}: Additional variant generation failed: {e}")
            return []

    def apply(self, operator_input: Dict[str, Any]) -> List[str]:
        """
        Generate text variants using POS-aware antonym replacement.
        
        This method:
        1. Validates input format and extracts parent data
        2. Detects POS tags and generates antonyms
        3. Creates variants with antonym substitutions
        4. Returns variants if different from original
        
        Args:
            operator_input (Dict[str, Any]): Operator input containing:
                - 'parent_data': Enriched parent genome dictionary containing:
                    - 'prompt': Original prompt text to process with antonym replacement
                    - 'generated_text': Generated output from the prompt (optional)
                    - 'scores': Moderation scores dictionary
                    - 'north_star_score': Primary optimization metric score
                - 'max_variants': Maximum number of variants to generate
                
        Returns:
            List[str]: List containing antonym-replaced prompt variants (or original if failed)
        """
        try:
            # Validate input format
            if not isinstance(operator_input, dict):
                self.logger.error(f"{self.name}: Input must be a dictionary")
                return []
            
            # Extract parent data
            parent_data = operator_input.get("parent_data", {})
            
            if not isinstance(parent_data, dict):
                self.logger.error(f"{self.name}: parent_data must be a dictionary")
                return []
            
            # Extract prompt from parent data
            text = parent_data.get("prompt", "")
            
            # Handle edge cases
            if not text or not text.strip():
                self.logger.debug(f"{self.name}: Empty input, returning as-is")
                return [text]
            
            # Detect and organize POS tags
            detected_pos = self._detect_and_organize_pos(text)
            
            if not detected_pos:
                self.logger.warning(f"{self.name}: No POS detected in text: '{text}'")
                return [text]
            
            # Select POS types to process
            selected_pos = self._select_pos_types(detected_pos)
            
            if not selected_pos:
                self.logger.warning(f"{self.name}: No POS selected from detected POS")
                return [text]
            self.logger.info(f"{self.name}: POS analysis complete - {len(detected_pos)} types detected, {len(selected_pos)} selected")
            
            # Generate antonyms using LLM
            antonyms_by_pos = self._generate_antonyms_for_selected_pos(detected_pos, selected_pos, text)
            
            if not antonyms_by_pos:
                self.logger.warning(f"{self.name}: No antonyms generated, returning original text")
                return [text]
            
            self.logger.info(f"{self.name}: Generated antonyms for {len(antonyms_by_pos)} POS types")
            
            # Generate text variants
            variants = self._generate_text_variants(text, detected_pos, antonyms_by_pos)
            
            if not variants:
                self.logger.warning(f"{self.name}: No variants generated, returning original text")
                return [text]
            
            self.logger.info(f"{self.name}: Generated {len(variants)} text variants")
            
            return variants
            
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed: {e}")
            return [text]

    def get_pos_info(self, text: str) -> Dict[str, Any]:
        """
        Helper method to get detailed POS information for a text.
        Useful for debugging and validation.
        
        Args:
            text: Input text
            
        Returns:
            Detailed POS information dictionary
        """
        detected_pos = self._detect_and_organize_pos(text)
        selected_pos = self._select_pos_types(detected_pos)
        
        return {
            'text': text,
            'text_length': len(text),
            'num_POS_tags_requested': self.num_POS_tags,
            'detected_pos_types': {
                pos_tag: {
                    'description': self.POS_DESCRIPTIONS[pos_tag],
                    'word_count': len(words),
                    'words': [w.word for w in words],
                    'positions': [(w.start, w.end) for w in words]
                }
                for pos_tag, words in detected_pos.items()
            },
            'selected_pos_types': selected_pos,
            'comprehensive_coverage': len(detected_pos),
            'selection_coverage': len(selected_pos)
        }
