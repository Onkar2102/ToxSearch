"""
Combined back_translation_operators.py

This module contains all classic (Helsinki-NLP) back-translation operators for the evolutionary algorithm.
Supported languages: Hindi, French, German, Japanese, Chinese
"""


# Standalone base and helpers
import torch
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from .VariationOperators import VariationOperator
from .operator_helpers import limit_variants
import logging
load_dotenv()
get_logger = logging.getLogger

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
        self.logger = get_logger(self.name)
        self.en_xx = None
        self.xx_en = None
        try:
            for model_id in (en_to_lang_repo, lang_to_en_repo):
                try:
                    snapshot_download(model_id, local_files_only=True)
                except Exception:
                    snapshot_download(model_id, local_files_only=False, resume_download=True)
            en_xx_model = AutoModelForSeq2SeqLM.from_pretrained(en_to_lang_repo, local_files_only=True)
            en_xx_tokenizer = AutoTokenizer.from_pretrained(en_to_lang_repo, local_files_only=True)
            self.en_xx = pipeline(pipeline_task_en_to_lang, model=en_xx_model, tokenizer=en_xx_tokenizer)
            xx_en_model = AutoModelForSeq2SeqLM.from_pretrained(lang_to_en_repo, local_files_only=True)
            xx_en_tokenizer = AutoTokenizer.from_pretrained(lang_to_en_repo, local_files_only=True)
            self.xx_en = pipeline(pipeline_task_lang_to_en, model=xx_en_model, tokenizer=xx_en_tokenizer)
        except Exception as e:
            self.en_xx = None
            self.xx_en = None

    def apply(self, text: str) -> List[str]:
        self._last_input = text
        self._last_intermediate = None
        self._last_final = None
        if self.en_xx is None or self.xx_en is None:
            self._last_intermediate = None
            self._last_final = text
            return [text]
        variants = set()
        attempts = 0
        original_normalized = text.strip().lower()
        last_lang_text = None
        while len(variants) < 4 and attempts < 10:
            try:
                lang_text = self.en_xx(text, max_length=1024)[0]['translation_text']
                last_lang_text = lang_text
                english = self.xx_en(lang_text, max_length=1024, do_sample=True, top_k=50)[0]['translation_text']
                cleaned = english.strip()
                normalized = cleaned.lower()
                if normalized and normalized != original_normalized and normalized not in variants:
                    variants.add(normalized)
            except Exception:
                pass
            attempts += 1
        result_variants = list({v.strip() for v in variants}) if variants else [text]
        limited_variants = limit_variants(result_variants, max_variants=3)
        self._last_intermediate = last_lang_text
        self._last_final = limited_variants[0] if limited_variants else text
        return limited_variants

class BackTranslationHIOperator(_GenericBackTranslationOperator):
    """Hindi back-translation operator."""
    def __init__(self, log_file=None):
        super().__init__(
            name="BackTranslation_HI",
            lang_code="hi",
            en_to_lang_repo="Helsinki-NLP/opus-mt-en-hi",
            lang_to_en_repo="Helsinki-NLP/opus-mt-hi-en",
            pipeline_task_en_to_lang="translation_en_to_hi",
            pipeline_task_lang_to_en="translation_hi_to_en",
            description_suffix="Hindi (Helsinki-NLP)",
            log_file=log_file,
        )

class BackTranslationFROperator(_GenericBackTranslationOperator):
    """French back-translation operator."""
    def __init__(self, log_file=None):
        super().__init__(
            name="BackTranslation_FR",
            lang_code="fr",
            en_to_lang_repo="Helsinki-NLP/opus-mt-en-fr",
            lang_to_en_repo="Helsinki-NLP/opus-mt-fr-en",
            pipeline_task_en_to_lang="translation_en_to_fr",
            pipeline_task_lang_to_en="translation_fr_to_en",
            description_suffix="French (Helsinki-NLP)",
            log_file=log_file,
        )

class BackTranslationDEOperator(_GenericBackTranslationOperator):
    """German back-translation operator."""
    def __init__(self, log_file=None):
        super().__init__(
            name="BackTranslation_DE",
            lang_code="de",
            en_to_lang_repo="Helsinki-NLP/opus-mt-en-de",
            lang_to_en_repo="Helsinki-NLP/opus-mt-de-en",
            pipeline_task_en_to_lang="translation_en_to_de",
            pipeline_task_lang_to_en="translation_de_to_en",
            description_suffix="German (Helsinki-NLP)",
            log_file=log_file,
        )

class BackTranslationJAOperator(_GenericBackTranslationOperator):
    """Japanese back-translation operator."""
    def __init__(self, log_file=None):
        super().__init__(
            name="BackTranslation_JA",
            lang_code="ja",
            en_to_lang_repo="Helsinki-NLP/opus-mt-en-ja",
            lang_to_en_repo="Helsinki-NLP/opus-mt-ja-en",
            pipeline_task_en_to_lang="translation_en_to_ja",
            pipeline_task_lang_to_en="translation_ja_to_en",
            description_suffix="Japanese (Helsinki-NLP)",
            log_file=log_file,
        )

class BackTranslationZHOperator(_GenericBackTranslationOperator):
    """Chinese back-translation operator."""
    def __init__(self, log_file=None):
        super().__init__(
            name="BackTranslation_ZH",
            lang_code="zh",
            en_to_lang_repo="Helsinki-NLP/opus-mt-en-zh",
            lang_to_en_repo="Helsinki-NLP/opus-mt-zh-en",
            pipeline_task_en_to_lang="translation_en_to_zh",
            pipeline_task_lang_to_en="translation_zh_to_en",
            description_suffix="Chinese (Helsinki-NLP)",
            log_file=log_file,
        )
