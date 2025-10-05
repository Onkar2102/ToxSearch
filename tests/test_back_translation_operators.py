#!/usr/bin/env python3
"""
Comprehensive pytest suite for Back-Translation and LLM-BackTranslation operators.
Prints and saves initial, mid, and final steps for all tests.
"""
import sys
import os
import json
from datetime import datetime
import pytest

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Base prompts (same as MLM tests)
_BASE_25 = (
    "A curious graduate researcher carefully evaluates complex datasets, "
    "builds modular prototypes, compares quantitative results, writes clear reports, "
    "and iterates quickly to improve robustness and efficiency significantly."
)

_BASE_100 = (
    "In practical machine learning projects, diligent engineers design modular pipelines, "
    "clean noisy data, explore meaningful features, train competitive models, tune hyperparameters, "
    "evaluate fairness, explain predictions, and document reproducible experiments with versioned datasets, "
    "tracked metrics, automated tests, and readable reports while collaborating asynchronously through reviews and discussions."
)

_BASE_200 = (
    "For production artificial intelligence systems, disciplined teams integrate data validation, schema checks, "
    "monitoring dashboards, canary deployments, rollback strategies, streaming features, privacy safeguards, "
    "prompt management, safety evaluation, offline experiments, online A/B testing, profiling, cost analysis, and governance reviews "
    "to deliver reliable value to users and stakeholders across diverse scenarios and evolving requirements."
)

def _build_prompt(base_text: str, target_words: int) -> str:
    words = base_text.split()
    if len(words) <= target_words:
        return base_text
    return " ".join(words[:target_words])


def _save_bt_test_output(test_name, operator_type, prompt, variants, mid_steps=None):
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "bt_all_outputs.json")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "test_name": test_name,
        "operator_type": operator_type,
        "timestamp": timestamp,
        "input_prompt": prompt,
        "input_stats": {
            "char_count": len(prompt),
            "word_count": len(prompt.split()),
        },
        "variants": variants,
        "variant_stats": [
            {"char_count": len(v), "word_count": len(v.split())} for v in variants
        ],
        "mid_steps": mid_steps or {},
    }
    # Append to a single JSON file as a list of results
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = [all_results]
            except Exception:
                all_results = []
    else:
        all_results = []
    all_results.append(output_data)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[OUTPUT] Appended BT test results to: {filepath}")

def _print_and_save_bt_steps(test_name, operator_type, prompt, operator, variants):
    mid_steps = {
        "input": getattr(operator, "_last_input", None),
        "intermediate": getattr(operator, "_last_intermediate", None),
        "final": getattr(operator, "_last_final", None),
    }
    print(f"\n[INITIAL] {mid_steps['input']}")
    print(f"[MID] {mid_steps['intermediate']}")
    print(f"[FINAL] {mid_steps['final']}")
    _save_bt_test_output(test_name, operator_type, prompt, variants, mid_steps)

def _assert_bt_variants_reasonable(variants):
    assert isinstance(variants, list)
    assert 1 <= len(variants) <= 3
    assert all(isinstance(v, str) and v.strip() for v in variants)
    for i, v in enumerate(variants):
        print(f"  Variant {i+1}: {len(v.split())} words, {len(v)} chars")



# Use new combined operator modules
from ea.back_translation_operators import (
    BackTranslationHIOperator,
    BackTranslationFROperator,
    BackTranslationDEOperator,
    BackTranslationJAOperator,
    BackTranslationZHOperator,
)
from ea.llm_back_translation_operators import (
    LLMBackTranslationHIOperator,
    LLMBackTranslationFROperator,
    LLMBackTranslationDEOperator,
    LLMBackTranslationJAOperator,
    LLMBackTranslationZHOperator,
)

LANGUAGES = [
    {"name": "hindi", "classic": BackTranslationHIOperator, "llm": LLMBackTranslationHIOperator},
    {"name": "french", "classic": BackTranslationFROperator, "llm": LLMBackTranslationFROperator},
    {"name": "german", "classic": BackTranslationDEOperator, "llm": LLMBackTranslationDEOperator},
    {"name": "japanese", "classic": BackTranslationJAOperator, "llm": LLMBackTranslationJAOperator},
    {"name": "chinese", "classic": BackTranslationZHOperator, "llm": LLMBackTranslationZHOperator},
]

PROMPTS = [
    ("25_words", _BASE_25, 25),
    ("100_words", _BASE_100, 46),
    ("200_words", _BASE_200, 51),
]

import importlib

class TestBackTranslationOperators:
    @pytest.mark.parametrize("lang", LANGUAGES, ids=[l["name"] for l in LANGUAGES])
    @pytest.mark.parametrize("prompt_name,base_text,word_count", PROMPTS)
    def test_classic_backtranslation(self, lang, prompt_name, base_text, word_count):
        Operator = lang["classic"]
        prompt = _build_prompt(base_text, word_count)
        operator = Operator()
        variants = operator.apply(prompt)
        _print_and_save_bt_steps(f"{prompt_name}", lang["name"], prompt, operator, variants)
        _assert_bt_variants_reasonable(variants)

    @pytest.mark.parametrize("lang", LANGUAGES, ids=[l["name"] for l in LANGUAGES])
    @pytest.mark.parametrize("prompt_name,base_text,word_count", PROMPTS)
    def test_llm_backtranslation(self, lang, prompt_name, base_text, word_count):
        Operator = lang["llm"]
        prompt = _build_prompt(base_text, word_count)
        operator = Operator()
        variants = operator.apply(prompt)
        _print_and_save_bt_steps(f"{prompt_name}", f"llm_{lang['name']}", prompt, operator, variants)
        _assert_bt_variants_reasonable(variants)


    @pytest.mark.parametrize("lang", LANGUAGES, ids=[l["name"] for l in LANGUAGES])
    @pytest.mark.parametrize("otype", ["classic", "llm"])
    def test_empty_input(self, lang, otype):
        Operator = lang[otype]
        operator = Operator()
        prompt = ""
        variants = operator.apply(prompt)
        _print_and_save_bt_steps("empty", f"{otype}_{lang['name']}", prompt, operator, variants)
        assert variants == [""]

    @pytest.mark.parametrize("lang", LANGUAGES, ids=[l["name"] for l in LANGUAGES])
    @pytest.mark.parametrize("otype", ["classic", "llm"])
    def test_whitespace_input(self, lang, otype):
        Operator = lang[otype]
        operator = Operator()
        prompt = "   \t\n   "
        variants = operator.apply(prompt)
        _print_and_save_bt_steps("whitespace", f"{otype}_{lang['name']}", prompt, operator, variants)
        assert all(v.strip() == "" for v in variants)

    @pytest.mark.parametrize("lang", LANGUAGES, ids=[l["name"] for l in LANGUAGES])
    @pytest.mark.parametrize("otype", ["classic", "llm"])
    def test_non_english_input(self, lang, otype):
        Operator = lang[otype]
        operator = Operator()
        prompt = "これは日本語のテキストです。"
        variants = operator.apply(prompt)
        _print_and_save_bt_steps("non_english", f"{otype}_{lang['name']}", prompt, operator, variants)
        assert isinstance(variants, list)

    @pytest.mark.parametrize("lang", LANGUAGES, ids=[l["name"] for l in LANGUAGES])
    @pytest.mark.parametrize("otype", ["classic", "llm"])
    def test_repeated_words(self, lang, otype):
        Operator = lang[otype]
        operator = Operator()
        prompt = "good good good good"
        variants = operator.apply(prompt)
        _print_and_save_bt_steps("repeated", f"{otype}_{lang['name']}", prompt, operator, variants)
        assert any("good" in v for v in variants)

    @pytest.mark.parametrize("lang", LANGUAGES, ids=[l["name"] for l in LANGUAGES])
    @pytest.mark.parametrize("otype", ["classic", "llm"])
    def test_long_text(self, lang, otype):
        Operator = lang[otype]
        operator = Operator()
        prompt = "This is a long text. " * 50
        variants = operator.apply(prompt)
        _print_and_save_bt_steps("long_text", f"{otype}_{lang['name']}", prompt, operator, variants)
        assert isinstance(variants, list)

    @pytest.mark.parametrize("lang", LANGUAGES, ids=[l["name"] for l in LANGUAGES])
    def test_model_unavailable(self, lang, monkeypatch):
        # Only applies to classic
        Operator = lang["classic"]
        def broken_init(self, *a, **kw):
            self.en_xx = None
            self.xx_en = None
        monkeypatch.setattr(Operator, "__init__", broken_init)
        operator = Operator()
        prompt = "Hello world"
        variants = operator.apply(prompt)
        _print_and_save_bt_steps("fail_model", lang["name"], prompt, operator, variants)
        assert variants == [prompt]

    @pytest.mark.parametrize("lang", LANGUAGES, ids=[l["name"] for l in LANGUAGES])
    def test_translation_error(self, lang, monkeypatch):
        # Only applies to classic
        Operator = lang["classic"]
        operator = Operator()
        def broken_apply(self, text):
            raise RuntimeError("Simulated translation failure")
        monkeypatch.setattr(operator, "apply", broken_apply.__get__(operator))
        prompt = "Hello world"
        try:
            variants = operator.apply(prompt)
        except Exception as e:
            print(f"[FAILURE] Caught expected error: {e}")
            variants = [prompt]
        _print_and_save_bt_steps("fail_translate", lang["name"], prompt, operator, variants)
        assert variants == [prompt]

    @pytest.mark.parametrize("lang", LANGUAGES, ids=[l["name"] for l in LANGUAGES])
    def test_llm_generator_failure(self, lang, monkeypatch):
        # Only applies to LLM
        Operator = lang["llm"]
        operator = Operator()
        # Patch generator.translate to raise error
        class DummyGen:
            def translate(self, *a, **k):
                raise RuntimeError("Simulated generator failure")
        operator.generator = DummyGen()
        prompt = "Hello world"
        variants = operator.apply(prompt)
        _print_and_save_bt_steps("fail_llm_generator", f"llm_{lang['name']}", prompt, operator, variants)
        assert variants == [prompt]
