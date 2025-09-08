#!/usr/bin/env python3
"""
download_models.py

Idempotent multi-model downloader & local loader for Hugging Face models.
- Downloads into a project-local folder (default: ./models)
- Supports aliases, --all / --only selection
- Always loads from disk after download

Usage examples:
    python download_models.py --all
    python download_models.py --only llama3.2-3b-instruct --target-dir models
    python download_models.py --only llama3.2-3b-instruct --revision main

Then, in your code:
    from download_models import load_local_model
    tok, model = load_local_model("llama3.2-3b-instruct")
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1) Registry: alias -> HF repo
# Extend as needed for your project
# -----------------------------
MODEL_REGISTRY: Dict[str, str] = {
    # Meta LLaMA 3.2
    "llama3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2-8b-instruct": "meta-llama/Llama-3.2-8B-Instruct",
    # Add more here as needed:
    # "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    # "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}

# -----------------------------
# 2) Defaults
# -----------------------------
DEFAULT_TARGET_DIR = Path("models")
REGISTRY_FILE = Path("models_registry.json")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def local_model_dir(target_dir: Path, alias: str) -> Path:
    # E.g., models/llama3.2-3b-instruct
    return target_dir / alias


def model_already_present(path: Path) -> bool:
    # Heuristic: presence of tokenizer + model files
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    return path.exists() and any((path / r).exists() for r in required)


def download_one(
    alias: str,
    repo_id: str,
    target_dir: Path,
    revision: Optional[str] = None,
) -> Path:
    out_dir = local_model_dir(target_dir, alias)
    ensure_dir(out_dir)

    if model_already_present(out_dir):
        print(f"[skip] {alias} already present at {out_dir}")
        return out_dir

    print(f"[download] {alias} ← {repo_id} -> {out_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,  # real files
        resume_download=True,          # idempotent
        revision=revision,             # optional branch/tag/commit
        # You can further filter files here if you want:
        # allow_patterns=["*.json","*.safetensors","tokenizer*","merges.txt","*.model"]
    )
    print(f"[ok] Saved {alias} at {out_dir}")
    return out_dir


def write_registry(registry_path: Path, mapping: Dict[str, str]) -> None:
    # Merge with existing (idempotent)
    existing = {}
    if registry_path.exists():
        try:
            existing = json.loads(registry_path.read_text())
        except Exception:
            pass
    existing.update(mapping)
    tmp = registry_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, indent=2))
    tmp.replace(registry_path)
    print(f"[registry] Updated {registry_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download HF models locally and load from disk.")
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true", help="Download all models in registry.")
    sel.add_argument("--only", nargs="+", help="Download only these aliases (space separated).")
    p.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR), help="Where to store models (default: ./models)")
    p.add_argument("--revision", default=None, help="Optional HF revision (branch/tag/commit) for all downloads.")
    p.add_argument("--login", action="store_true", help="Run huggingface login first (useful for gated models).")
    return p.parse_args()


def maybe_login(do_login: bool) -> None:
    if do_login:
        try:
            print("[auth] Logging in to Hugging Face Hub…")
            login()  # will prompt for token if not set
        except Exception as e:
            print(f"[warn] login failed or skipped: {e}")


def main() -> None:
    args = parse_args()
    maybe_login(args.login)

    target_dir = Path(args.target_dir).resolve()
    ensure_dir(target_dir)

    if args.all:
        to_get = list(MODEL_REGISTRY.keys())
    else:
        to_get = args.only

    # Validate aliases
    missing = [a for a in to_get if a not in MODEL_REGISTRY]
    if missing:
        print(f"[error] Unknown model alias(es): {missing}")
        print("Available aliases:", ", ".join(MODEL_REGISTRY.keys()))
        sys.exit(1)

    # Download loop
    resolved = {}
    for alias in to_get:
        repo_id = MODEL_REGISTRY[alias]
        path = download_one(alias, repo_id, target_dir, revision=args.revision)
        resolved[alias] = str(path)

    # Write/update registry mapping alias -> local path
    write_registry(REGISTRY_FILE, resolved)

    print("\nDone. Local models:")
    for k, v in resolved.items():
        print(f"  - {k}: {v}")


# -----------------------------
# 3) Local loading helpers
# -----------------------------
def _read_registry() -> Dict[str, str]:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}


def load_local_model(alias: str, device_map: str = "auto"):
    """
    Load a model previously downloaded by this script.
    Always uses local disk (no network).
    """
    reg = _read_registry()
    if alias not in reg:
        raise ValueError(
            f"Alias '{alias}' not found in {REGISTRY_FILE}. "
            f"Run: python download_models.py --only {alias}"
        )
    local_path = reg[alias]
    # Load tokenizer & model from disk
    tok = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        device_map=device_map,
        torch_dtype="auto",
    )
    return tok, model


if __name__ == "__main__":
    main()