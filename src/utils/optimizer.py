# System Performance Optimizer for Evolutionary Text Generation
#
# Provides system info and config optimization for local inference.

import psutil
import torch
import yaml
import json
from pathlib import Path
from typing import Dict

def get_system_info() -> Dict:
    """
    Get comprehensive system information for local inference optimization.
    Returns:
        Dict: System information including CPU, memory, disk, and accelerator details
    """
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/')._asdict(),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }
    # Optionally add more hardware info here if needed
    return info


def optimize_config_for_local() -> Dict:
    """Generate optimized configuration for local inference"""
    system_info = get_system_info()
    config = {
        "llama": {
            "provider": "huggingface",
            "name": "meta-llama/Llama-3.2-3B-instruct",
            "strategy": "local",
            "task_type": "text-generation",
            "generation_args": {
                "max_new_tokens": 512,
                "do_sample": False,
                "temperature": 0.8,
                "top_p": 0.9,
                "num_return_sequences": 1,
                "repetition_penalty": 1.1,
                "pad_token_id": 128001
            },
            "prompt_template": {
                "style": "chat",
                "user_prefix": "User:",
                "assistant_prefix": "System:",
                "format": "{{user_prefix}} {{prompt}}\n{{assistant_prefix}}"
            }
        }
    }
    if system_info["memory_available_gb"] < 8:
        config["llama"]["generation_args"]["max_new_tokens"] = 256
    elif system_info["memory_available_gb"] > 16:
        config["llama"]["generation_args"]["max_new_tokens"] = 1024
    return config

def main():
    import argparse
    parser = argparse.ArgumentParser(description="System Performance Optimizer")
    parser.add_argument("--system-info", action="store_true", help="Show system information")
    parser.add_argument("--optimize-config", action="store_true", help="Generate optimized config")
    parser.add_argument("--all", action="store_true", help="Run all optimizations in sequence")
    args = parser.parse_args()
    if args.all:
        print("=== Running all optimizations in sequence ===\n")
        print("1. System Information:")
        print("=" * 50)
        info = get_system_info()
        print(json.dumps(info, indent=2))
        print("\n")
        print("2. Optimizing Configuration:")
        print("=" * 50)
        config = optimize_config_for_local()
        config_path = Path("../config/modelConfig_llamacpp.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Optimized configuration saved to {config_path}")
        print("\n")
        print("Optimization complete!")
    elif args.system_info:
        info = get_system_info()
        print(json.dumps(info, indent=2))
    elif args.optimize_config:
        config = optimize_config_for_local()
        config_path = Path("../config/modelConfig_llamacpp.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Optimized configuration saved to {config_path}")
    else:
        print("System Optimizer - choose an option:")
        print("  --system-info: Show system information")
        print("  --optimize-config: Generate optimized configuration")
        print("  --all: Run all optimizations in sequence")

if __name__ == "__main__":
    main()
