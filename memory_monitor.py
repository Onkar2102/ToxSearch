#!/usr/bin/env python3
"""
Memory Monitor for Enhanced Memory Management

This script monitors memory usage and shows the improvements from the new memory management system.
"""

import os
import sys
import time
import psutil
import torch
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def monitor_memory():
    """Monitor memory usage in real-time"""
    print("🧠 Enhanced Memory Management Monitor")
    print("=" * 50)
    print("Memory threshold: 12GB (increased from 4GB)")
    print("Adaptive batch sizing: Enabled")
    print("Model cache clearing: After each generation")
    print("=" * 50)
    
    try:
        from gne.LLaMaTextGenerator import LlaMaTextGenerator
        
        # Initialize generator to get memory stats
        generator = LlaMaTextGenerator()
        
        while True:
            # Get system memory
            memory = psutil.virtual_memory()
            
            # Get generator memory stats
            try:
                stats = generator.get_current_memory_stats()
                
                print(f"\n📊 Memory Status - {time.strftime('%H:%M:%S')}")
                print(f"System Memory: {memory.used / (1024**3):.1f}GB used, {memory.available / (1024**3):.1f}GB available ({memory.percent:.1f}%)")
                print(f"Process Memory: {stats['total_memory_gb']:.2f}GB ({stats['memory_usage_percent']:.1f}% of {stats['max_memory_limit_gb']:.1f}GB limit)")
                print(f"CPU Memory: {stats['cpu_memory_gb']:.2f}GB")
                
                if torch.cuda.is_available():
                    print(f"GPU Memory: {stats['gpu_memory_gb']:.2f}GB")
                
                # Memory status indicators
                if stats['memory_usage_percent'] > 90:
                    status = "🔴 CRITICAL"
                elif stats['memory_usage_percent'] > 75:
                    status = "🟡 WARNING"
                elif stats['memory_usage_percent'] > 50:
                    status = "🟢 MODERATE"
                else:
                    status = "🟢 EXCELLENT"
                
                print(f"Memory Status: {status}")
                print(f"Adaptive Batch Sizing: {'✅ Enabled' if stats['adaptive_batch_sizing'] else '❌ Disabled'}")
                print(f"Memory Cleanup: {'✅ Enabled' if stats['memory_cleanup_enabled'] else '❌ Disabled'}")
                
            except Exception as e:
                print(f"Error getting generator stats: {e}")
                print(f"System Memory: {memory.used / (1024**3):.1f}GB used, {memory.available / (1024**3):.1f}GB available")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    monitor_memory() 