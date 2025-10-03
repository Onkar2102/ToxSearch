#!/usr/bin/env python3
"""
M3 Mac Performance Optimizer for Evolutionary Text Generation

Author: Onkar Shelar os9660@rit.edu

This utility helps optimize performance on Apple M3 Macs by:
1. Detecting optimal batch sizes based on available memory
2. Monitoring GPU/CPU usage during runs
3. Providing performance recommendations
4. Tuning configuration automatically
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
    """Get comprehensive system information for M3 Mac"""
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

def estimate_optimal_batch_size(model_name: str = "meta-llama/Llama-3.2-3B-instruct") -> int:
    """Estimate optimal batch size based on available memory and model size"""
    memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Model size estimates (in GB)
    model_sizes = {
        "meta-llama/Llama-3.2-3B-instruct": 6.0,  # ~6GB in fp16
        "meta-llama/Llama-3.2-1B-instruct": 2.0,   # ~2GB in fp16
    }
    
    model_size = model_sizes.get(model_name, 6.0)  # Default to 3B size
    
    # Reserve memory for system and other processes (50% of available)
    usable_memory = memory_gb * 0.5
    
    # Estimate memory per sample (model + activations + gradients)
    # For inference only, roughly 2x model size for batch of 1
    memory_per_sample = model_size * 0.3  # Conservative estimate
    
    if usable_memory < model_size:
        return 1  # Minimum batch size
    
    estimated_batch_size = int((usable_memory - model_size) / memory_per_sample)
    
    # Clamp to reasonable bounds
    return max(1, min(estimated_batch_size, 16))

def benchmark_generation_speed(batch_sizes: list = [1, 2, 4, 8]) -> Dict:
    """Benchmark text generation speed with different batch sizes"""
    try:
        from gne import get_LLaMaTextGenerator
        LlaMaTextGenerator = get_LLaMaTextGenerator()
    except ImportError as e:
        print(f"Warning: Could not import LLaMaTextGenerator: {e}")
        print("Skipping benchmark - this requires the full project setup")
        return {"error": "Import failed - run from project root directory"}
    
    results = {}
    test_prompts = ["Tell me about artificial intelligence." * i for i in range(1, 6)]
    
    print("Benchmarking batch sizes... This may take a few minutes.")
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        try:
            # Update config temporarily
            config_path = "../config/modelConfig.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            original_batch_size = config['llama']['max_batch_size']
            config['llama']['max_batch_size'] = batch_size
            
            # Save temporary config
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Initialize generator
            generator = LlaMaTextGenerator(config_path="../config/modelConfig.yaml")
            
            # Benchmark
            start_time = time.time()
            responses = generator.generate_response_batch(test_prompts[:batch_size])
            end_time = time.time()
            
            tokens_generated = sum(len(r.split()) for r in responses)
            time_taken = end_time - start_time
            
            results[batch_size] = {
                "time_seconds": time_taken,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_generated / time_taken,
                "prompts_per_second": len(responses) / time_taken
            }
            
            # Restore original config
            config['llama']['max_batch_size'] = original_batch_size
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                
        except Exception as e:
            print(f"Benchmark failed for batch size {batch_size}: {e}")
            results[batch_size] = {"error": str(e)}
    
    return results

def optimize_config_for_m3() -> Dict:
    """Generate optimized configuration for M3 Mac"""
    system_info = get_system_info()
    optimal_batch_size = estimate_optimal_batch_size()
    
    config = {
        "llama": {
            "provider": "huggingface",
            "name": "meta-llama/Llama-3.2-3B-instruct",
            "strategy": "local",
            "task_type": "text-generation",
            "max_batch_size": optimal_batch_size,
            "generation_args": {
                "max_new_tokens": 512,  # Balanced speed/quality
                "do_sample": False,
                "temperature": 0.8,
                "top_k": 40,
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
        config["llama"]["max_batch_size"] = 2
        config["llama"]["generation_args"]["max_new_tokens"] = 256
    elif system_info["memory_available_gb"] > 16:
        # High memory - aggressive settings
        config["llama"]["max_batch_size"] = min(optimal_batch_size * 2, 16)
        config["llama"]["generation_args"]["max_new_tokens"] = 1024
    
    return config

def monitor_performance(duration_seconds: int = 60) -> Dict:
    """Monitor system performance for a given duration"""
    print(f"Monitoring system performance for {duration_seconds} seconds...")
    
    measurements = []
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        measurement = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        }
        measurements.append(measurement)
        time.sleep(1)
    
    # Calculate averages
    avg_cpu = sum(m["cpu_percent"] for m in measurements) / len(measurements)
    avg_memory = sum(m["memory_percent"] for m in measurements) / len(measurements)
    min_memory_available = min(m["memory_available_gb"] for m in measurements)
    
    return {
        "duration_seconds": duration_seconds,
        "measurements": measurements,
        "average_cpu_percent": avg_cpu,
        "average_memory_percent": avg_memory,
        "minimum_memory_available_gb": min_memory_available,
        "peak_memory_usage_gb": psutil.virtual_memory().total / (1024**3) - min_memory_available
    }

def generate_performance_report() -> str:
    """Generate a comprehensive performance report"""
    system_info = get_system_info()
    optimal_batch = estimate_optimal_batch_size()
    
    report = f"""
# M3 Mac Performance Report

## System Information
- Chip: {system_info.get('chip', 'Unknown')}
- CPU Cores: {system_info['cpu_count']}
- Total Memory: {system_info['memory_total_gb']:.1f} GB
- Available Memory: {system_info['memory_available_gb']:.1f} GB
- Memory Usage: {system_info['memory_percent']:.1f}%
- MPS Available: {system_info['mps_available']}
- PyTorch Version: {system_info['torch_version']}

## Optimization Recommendations
- Recommended Batch Size: {optimal_batch}
- Model: meta-llama/Llama-3.2-3B-instruct (optimized for M3)
- Memory Strategy: Use fp16 precision with MPS backend
- Generation Strategy: Short sequences (512 tokens) for optimal throughput

## Performance Tips for M3 Mac
1. Use the optimized configuration generated by this script
2. Monitor memory usage during long runs
3. Consider smaller models (1B) if memory is constrained
4. Use batch processing for OpenAI API calls
5. Enable MPS backend for GPU acceleration

## Next Steps
1. Run `python src/utils/m3_optimizer.py --optimize-config` to update your config
2. Run `python src/utils/m3_optimizer.py --benchmark` to test performance
3. Start your evolution with `python src/main.py --generations 5`
"""
    
    return report

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="M3 Mac Performance Optimizer")
    parser.add_argument("--system-info", action="store_true", help="Show system information")
    parser.add_argument("--optimize-config", action="store_true", help="Generate optimized config")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark batch sizes")
    parser.add_argument("--monitor", type=int, default=60, help="Monitor performance for N seconds")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
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
        print(f"Recommended batch size: {config['llama']['max_batch_size']}")
        print("\n")
        
        # 3. Benchmark
        print("3. Benchmarking Performance:")
        print("=" * 50)
        results = benchmark_generation_speed()
        
        if "error" in results:
            print(f"Benchmark skipped: {results['error']}")
            print("Continuing with other optimizations...")
        else:
            print("Benchmark Results:")
            print(json.dumps(results, indent=2))
            
            # Find best batch size
            best_batch_size = 1
            best_speed = 0
            for batch_size, result in results.items():
                if "tokens_per_second" in result and result["tokens_per_second"] > best_speed:
                    best_speed = result["tokens_per_second"]
                    best_batch_size = batch_size
            print(f"\nRecommendation: Use batch size {best_batch_size} for best performance")
        print("\n")
        
        # 4. Monitor (shorter duration for all-in-one run)
        print("4. Performance Monitoring (30 seconds):")
        print("=" * 50)
        results = monitor_performance(30)
        print("Performance Monitoring Results:")
        print(json.dumps({k: v for k, v in results.items() if k != "measurements"}, indent=2))
        print("\n")
        
        # 5. Generate Report
        print("5. Generating Performance Report:")
        print("=" * 50)
        report = generate_performance_report()
        print(report)
        
        # Save report
        report_path = Path("outputs/m3_performance_report.md")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")
        
        print("\n=== All optimizations completed! ===")
        
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
        print(f"Recommended batch size: {config['llama']['max_batch_size']}")
    
    elif args.benchmark:
        results = benchmark_generation_speed()
        print("\nBenchmark Results:")
        print(json.dumps(results, indent=2))
        
        # Find best batch size
        best_batch_size = 1
        best_speed = 0
        for batch_size, result in results.items():
            if "tokens_per_second" in result and result["tokens_per_second"] > best_speed:
                best_speed = result["tokens_per_second"]
                best_batch_size = batch_size
        
        print(f"\nRecommendation: Use batch size {best_batch_size} for best performance")
    
    elif args.monitor:
        results = monitor_performance(args.monitor)
        print("\nPerformance Monitoring Results:")
        print(json.dumps({k: v for k, v in results.items() if k != "measurements"}, indent=2))
    
    elif args.report:
        report = generate_performance_report()
        print(report)
        
        # Save report
        report_path = Path("outputs/m3_performance_report.md")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")
    
    else:
        print("M3 Mac Optimizer - choose an option:")
        print("  --system-info: Show system information")
        print("  --optimize-config: Generate optimized configuration")
        print("  --benchmark: Test different batch sizes")
        print("  --monitor N: Monitor performance for N seconds")
        print("  --report: Generate comprehensive report")

if __name__ == "__main__":
    main() 