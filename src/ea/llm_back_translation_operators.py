"""
llm_back_translation_operators.py

LLM-based back-translation operators for text mutation through language round-trips.

This module implements mutation operators that translate text from English to
various target languages and back to English, creating paraphrased variants
through the translation process. Each operator supports a specific language.

Supported languages: Hindi (HI), French (FR), German (DE), Japanese (JA), Chinese (ZH)

Process for each operator:
1. Translate English text to target language using LLaMA
2. Translate back from target language to English
3. Return the back-translated variant if different from original
"""


from .VariationOperators import VariationOperator
import logging
from typing import List, Dict, Any
get_logger = logging.getLogger

class _GenericLLMBackTranslationOperator(VariationOperator):
    """Generic LLaMA-based back-translation operator for text mutation."""
    def __init__(self, name: str, target_lang: str, target_lang_code: str, log_file=None, generator=None):
        super().__init__(name, "mutation", f"LLaMA-based EN→{target_lang_code.upper()}→EN back-translation.")
        self.logger = get_logger(self.name)
        self.target_lang = target_lang
        self.target_lang_code = target_lang_code
        self.generator = generator

    def apply(self, operator_input: Dict[str, Any]) -> List[str]:
        """Generate back-translated variant using LLaMA model."""
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
            
            if not text:
                self.logger.error(f"{self.name}: Parent data missing required 'prompt' field")
                return []
            
            # Store debug information
            self._last_input = text
            self._last_intermediate = None
            self._last_final = None
            
            # Perform back-translation
            inter = self.generator.translate(text, self.target_lang, "English")
            self._last_intermediate = inter
            
            if inter and inter != text:
                back_en = self.generator.translate(inter, "English", self.target_lang)
                cleaned = back_en.strip()
                self._last_final = cleaned
                if cleaned and cleaned.lower() != text.strip().lower():
                    self.logger.info(f"{self.name}: Generated back-translated variant")
                    return [cleaned]
            
            self._last_final = text
            self.logger.warning(f"{self.name}: Back-translation returned same or empty text, returning original")
            return [text]
            
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed with error: {e}")
            return []

class LLMBackTranslationHIOperator(_GenericLLMBackTranslationOperator):
    """LLaMA-based Hindi back-translation operator."""
    def __init__(self, log_file=None, generator=None):
        super().__init__(
            name="LLMBackTranslation_HI",
            target_lang="Hindi",
            target_lang_code="hi",
            log_file=log_file,
            generator=generator,
        )

class LLMBackTranslationFROperator(_GenericLLMBackTranslationOperator):
    """LLaMA-based French back-translation operator."""
    def __init__(self, log_file=None, generator=None):
        super().__init__(
            name="LLMBackTranslation_FR",
            target_lang="French",
            target_lang_code="fr",
            log_file=log_file,
            generator=generator,
        )

class LLMBackTranslationDEOperator(_GenericLLMBackTranslationOperator):
    """LLaMA-based German back-translation operator."""
    def __init__(self, log_file=None, generator=None):
        super().__init__(
            name="LLMBackTranslation_DE",
            target_lang="German",
            target_lang_code="de",
            log_file=log_file,
            generator=generator,
        )

class LLMBackTranslationJAOperator(_GenericLLMBackTranslationOperator):
    """LLaMA-based Japanese back-translation operator."""
    def __init__(self, log_file=None, generator=None):
        super().__init__(
            name="LLMBackTranslation_JA",
            target_lang="Japanese",
            target_lang_code="ja",
            log_file=log_file,
            generator=generator,
        )

class LLMBackTranslationZHOperator(_GenericLLMBackTranslationOperator):
    """LLaMA-based Chinese back-translation operator."""
    def __init__(self, log_file=None, generator=None):
        super().__init__(
            name="LLMBackTranslation_ZH",
            target_lang="Chinese",
            target_lang_code="zh",
            log_file=log_file,
            generator=generator,
        )
