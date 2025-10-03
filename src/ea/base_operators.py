"""
base_operators.py


This module contains base classes for generic text variation operators.
These base classes implement common functionality for operator families.

Classes:
    _GenericBackTranslationOperator: Base class for model-based back-translation operators
    _GenericLLMBackTranslationOperator: Base class for LLaMA-based back-translation operators

Version: 1.0
"""

import torch
from typing import List, Optional, Dict, Any, Tuple
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

try:
    from ea.VariationOperators import VariationOperator
except Exception:
    # Fallback for direct module execution without package context
    from VariationOperators import VariationOperator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()

load_dotenv()


class _GenericBackTranslationOperator(VariationOperator):
    """
    Generic EN→XX→EN back-translation operator.

    Loads translation pipelines for EN→XX and XX→EN using Helsinki-NLP models
    with local-first behavior and logs both the intermediate translated text and
    the final back-translation variants.

    Subclasses should pass appropriate language codes and human-readable names.
    """
    def __init__(self, name: str, lang_code: str, en_to_lang_repo: str, lang_to_en_repo: str, pipeline_task_en_to_lang: str, pipeline_task_lang_to_en: str, description_suffix: str, log_file=None):
        super().__init__(name, "mutation", f"Performs EN→{lang_code.upper()}→EN back-translation. {description_suffix}")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")

        self.en_xx = None
        self.xx_en = None

        try:
            # Ensure models are present locally; fallback to download
            for model_id in (en_to_lang_repo, lang_to_en_repo):
                try:
                    snapshot_download(model_id, local_files_only=True)
                except Exception:
                    self.logger.info(f"Model {model_id} not found in cache. Downloading...")
                    snapshot_download(model_id, local_files_only=False, resume_download=True)

            en_xx_model = AutoModelForSeq2SeqLM.from_pretrained(en_to_lang_repo, local_files_only=True)
            en_xx_tokenizer = AutoTokenizer.from_pretrained(en_to_lang_repo, local_files_only=True)
            self.en_xx = pipeline(pipeline_task_en_to_lang, model=en_xx_model, tokenizer=en_xx_tokenizer)

            xx_en_model = AutoModelForSeq2SeqLM.from_pretrained(lang_to_en_repo, local_files_only=True)
            xx_en_tokenizer = AutoTokenizer.from_pretrained(lang_to_en_repo, local_files_only=True)
            self.xx_en = pipeline(pipeline_task_lang_to_en, model=xx_en_model, tokenizer=xx_en_tokenizer)

            self.logger.info(f"Successfully initialized {self.name} with translation models")
        except Exception as e:
            self.logger.warning(f"Failed to initialize {self.name}: {e}. Operator will be disabled.")
            self.en_xx = None
            self.xx_en = None

    def apply(self, text: str) -> List[str]:
        if self.en_xx is None or self.xx_en is None:
            self.logger.warning(f"{self.name}: Translation models not available, returning original text")
            return [text]

        variants = set()
        attempts = 0
        original_normalized = text.strip().lower()
        while len(variants) < 4 and attempts < 10:
            try:
                # Translate EN -> XX and log the intermediate text
                lang_text = self.en_xx(text, max_length=1024)[0]['translation_text']
                if lang_text:
                    self.logger.debug(f"{self.name}: Intermediate translation: '{lang_text}'")

                # Translate XX -> EN with sampling for diversity
                english = self.xx_en(lang_text, max_length=1024, do_sample=True, top_k=50)[0]['translation_text']
                cleaned = english.strip()
                normalized = cleaned.lower()
                if normalized and normalized != original_normalized and normalized not in variants:
                    self.logger.debug(f"{self.name}: Back-translated to '{cleaned}'")
                    variants.add(normalized)
            except Exception as e:
                self.logger.error(f"[{self.name} error]: {e}")
            attempts += 1

        result_variants = list({v.strip() for v in variants}) if variants else [text]
        from .operator_helpers import limit_variants
        limited_variants = limit_variants(result_variants, max_variants=3)
        self.logger.debug(f"{self.name}: Generated {len(result_variants)} unique back-translations, limited to {len(limited_variants)} for: '{text[:60]}...'")
        return limited_variants


class _GenericLLMBackTranslationOperator(VariationOperator):
    """
    Generic LLaMA-based back-translation operator (EN → target_lang → EN).

    This operator uses the cached LLaMA generator from `get_generator()` and its
    task-specific translate() method to:
    1) Translate English to a target language using config templates
    2) Translate back to English with natural phrasing
    It logs the intermediate translated text and returns up to 3 unique variants.

    Subclasses should pass appropriate language names and codes.
    """
    def __init__(self, name: str, target_lang: str, target_lang_code: str, log_file=None):
        super().__init__(name, "mutation", f"LLaMA-based EN→{target_lang_code.upper()}→EN back-translation.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
        self.target_lang = target_lang
        self.target_lang_code = target_lang_code
        # Shared local LLaMA generator
        from .operator_helpers import get_generator
        self.generator = get_generator()

    def apply(self, text: str) -> List[str]:
        try:
            # Single back-translation attempt: EN → target → EN
            inter = self.generator.translate(text, self.target_lang, "English")
            if inter and inter != text:
                self.logger.debug(f"{self.name}: Intermediate {self.target_lang} translation: '{inter}'")
                
                # Translate back to English
                back_en = self.generator.translate(inter, "English", self.target_lang)
                cleaned = back_en.strip()
                
                if cleaned and cleaned.lower() != text.strip().lower():
                    self.logger.debug(f"{self.name}: Back-translated to '{cleaned}'")
                    return [cleaned]
        except Exception as e:
            self.logger.warning(f"{self.name}: LLaMA back-translation failed: {e}")
        
        # Fallback to original text if translation fails
        return [text]