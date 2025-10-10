#!/usr/bin/env python3
"""
M3 Mac Performance Optimizer for Evolutionary Text Generation

This utility provides system optimization and monitoring for Apple Silicon Macs,
specifically optimized for M3 chips running the evolutionary text generation pipeline.

Functionality:
1. Detect optimal settings based on available memory and MPS capabilities
2. Monitor GPU/CPU/memory usage during evolution runs
3. Provide performance recommendations for model configuration
4. Automatically tune YAML configuration for optimal Apple Silicon performance
5. Support for MPS (Metal Performance Shaders) acceleration
"""

import os
import sys

# Add the src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import psutil
import torch
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Tuple
import subprocess

def get_system_info() -> Dict:
    """
    Get comprehensive system information for Apple Silicon Macs.
    
    Collects hardware specifications, memory usage, CPU information,
    and PyTorch/MPS availability for performance optimization.
    
    Returns:
        Dict: System information including CPU, memory, disk, and Apple Silicon details
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
    
    # Try to get more specific Mac info
    try:
        result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Chip:' in line:
                    info['chip'] = line.split('Chip:')[1].strip()
                elif 'Total Number of Cores:' in line:
                    info['total_cores'] = line.split('Total Number of Cores:')[1].strip()
                elif 'Memory:' in line:
                    info['system_memory'] = line.split('Memory:')[1].strip()
    except Exception as e:
        # Silently ignore parsing errors for system info
        pass
    
    return info


def optimize_config_for_m3() -> Dict:
    """Generate optimized configuration for M3 Mac"""
    system_info = get_system_info()
    
    config = {
        "llama": {
            "provider": "huggingface",
            "name": "meta-llama/Llama-3.2-3B-instruct",
            "strategy": "local",
            "task_type": "text-generation",
            "generation_args": {
                "max_new_tokens": 512,  # Balanced speed/quality
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
    
    # Adjust based on available memory
    if system_info["memory_available_gb"] < 8:
        # Low memory - conservative settings
        config["llama"]["generation_args"]["max_new_tokens"] = 256
    elif system_info["memory_available_gb"] > 16:
        # High memory - aggressive settings
        config["llama"]["generation_args"]["max_new_tokens"] = 1024
    
    return config

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="M3 Mac Performance Optimizer")
    parser.add_argument("--system-info", action="store_true", help="Show system information")
    parser.add_argument("--optimize-config", action="store_true", help="Generate optimized config")
    parser.add_argument("--all", action="store_true", help="Run all optimizations in sequence")
    
    args = parser.parse_args()
    
    if args.all:
        print("=== Running all M3 optimizations in sequence ===\n")
        
        # 1. System Info
        print("1. System Information:")
        print("=" * 50)
        info = get_system_info()
        print(json.dumps(info, indent=2))
        print("\n")
        
        # 2. Optimize Config
        print("2. Optimizing Configuration:")
        print("=" * 50)
        config = optimize_config_for_m3()
        config_path = Path("../config/modelConfig.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Optimized configuration saved to {config_path}")
        print("\n")
        
        print("Optimization complete!")
        
    elif args.system_info:
        info = get_system_info()
        print(json.dumps(info, indent=2))
    
    elif args.optimize_config:
        config = optimize_config_for_m3()
        
        # Save optimized config
        config_path = Path("../config/modelConfig.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Optimized configuration saved to {config_path}")
    
    else:
        print("M3 Mac Optimizer - choose an option:")
        print("  --system-info: Show system information")
        print("  --optimize-config: Generate optimized configuration")
        print("  --all: Run all optimizations in sequence")

if __name__ == "__main__":
    main() 