# Performance Optimization Guide

## Overview

This guide explains the performance optimizations implemented for `llama_cpp_python` to reduce MacBook noise and improve efficiency across different devices.

## Issues Addressed

### 1. Excessive Resource Usage
- **Problem**: `n_gpu_layers: -1` was using ALL GPU layers, causing excessive resource consumption
- **Solution**: Device-specific GPU layer configuration with reasonable defaults

### 2. Device Detection Issues
- **Problem**: MPS detection on Linux systems
- **Solution**: Improved device detection logic with OS-specific checks

### 3. Missing Device-Specific Optimizations
- **Problem**: Same configuration for all devices
- **Solution**: Device-specific llama_cpp_python parameters

## Optimizations Implemented

### macOS (MPS/Metal Performance Shaders)
```yaml
mps:
  gpu_layers: 20          # Reasonable default (was -1)
  low_vram: false         # Not needed for MPS
  use_mmap: true          # Memory mapping for efficiency
  use_mlock: false        # Don't lock memory
  f16_kv: true           # Use half precision for speed
```

### NVIDIA CUDA
```yaml
cuda:
  gpu_layers: -1          # Use all layers for CUDA
  low_vram: false         # Adjust based on GPU memory
  use_mmap: true          # Memory mapping
  use_mlock: false        # Don't lock memory
  f16_kv: true           # Half precision for speed
```

### CPU
```yaml
cpu:
  gpu_layers: 0           # No GPU layers
  num_threads: null       # Auto-detect
  use_mmap: true         # Memory mapping
  use_mlock: false       # Don't lock memory
  f16_kv: false          # Use f32 for CPU stability
```

## Installation & Setup

### For macOS (Apple Silicon)
```bash
# Run the Metal setup script
./setup_macos_metal.sh
```

This script will:
1. Uninstall existing llama_cpp_python
2. Install with Metal support: `CMAKE_ARGS="-DGGML_METAL=on"`
3. Verify Metal acceleration is working

### For Linux (NVIDIA CUDA)
```bash
# Run the CUDA setup script
./setup_gpu.sh
```

## Performance Monitoring

### Real-time Monitoring
```bash
# Monitor performance continuously
python3 performance_monitor.py --continuous

# Single performance check
python3 performance_monitor.py
```

### Key Metrics to Watch
- **CPU Usage**: Should be < 80% under normal load
- **Memory Usage**: Should be < 85% to avoid swapping
- **GPU Utilization**: Monitor Metal GPU usage on macOS
- **Temperature**: Use Activity Monitor to check CPU/GPU temps

## Configuration Tuning

### Adjust GPU Layers
Edit `config/RGConfig.yaml` and `config/PGConfig.yaml`:

```yaml
device_config:
  mps:
    gpu_layers: 15  # Reduce for less GPU usage
    # or
    gpu_layers: 25  # Increase for more GPU usage
```

**Guidelines:**
- **10-15 layers**: Lower GPU usage, less noise, slower inference
- **20-25 layers**: Balanced performance and resource usage
- **30+ layers**: Maximum speed, higher resource usage

### Memory Management
```yaml
response_generator:
  max_memory_usage_gb: 16.0  # Reduce if you have less RAM
  enable_memory_cleanup: true
```

## Troubleshooting

### High CPU Usage
1. Increase `gpu_layers` to offload more work to GPU
2. Check if Metal/CUDA is properly installed
3. Monitor with `performance_monitor.py`

### High Memory Usage
1. Reduce `max_memory_usage_gb`
2. Enable `enable_memory_cleanup`
3. Reduce `gpu_layers` if GPU memory is limited

### Device Detection Issues
1. Check logs for device detection messages
2. Verify OS detection: `python3 -c "import platform; print(platform.system())"`
3. Force device in config: `preferred_device: "cpu"` or `"cuda"`

## Performance Benchmarks

### Before Optimization
- GPU layers: -1 (all layers)
- No device-specific tuning
- High resource usage
- MacBook fan noise

### After Optimization
- GPU layers: 20 (reasonable default)
- Device-specific parameters
- Balanced resource usage
- Reduced fan noise

## Advanced Configuration

### Custom Device Settings
You can override device detection by setting `preferred_device` in your config:

```yaml
device_config:
  preferred_device: "cpu"    # Force CPU usage
  # or
  preferred_device: "cuda"   # Force CUDA usage
  # or
  preferred_device: "mps"    # Force MPS usage
```

### Model-Specific Settings
Different models may require different settings:

```yaml
# For smaller models (7B)
mps:
  gpu_layers: 25

# For larger models (13B+)
mps:
  gpu_layers: 15
```

## Monitoring Commands

### Check Device Detection
```bash
source venv/bin/activate
python3 -c "
import platform
import torch
print('OS:', platform.system())
print('CUDA available:', torch.cuda.is_available())
print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'No MPS backend')
"
```

### Check llama_cpp_python Installation
```bash
python3 -c "
from llama_cpp import Llama
print('llama_cpp_python installed successfully')
"
```

### Monitor System Resources
```bash
# macOS
top -l 1 | grep -E "(CPU|PhysMem)"

# Linux
htop
```

## Expected Results

After implementing these optimizations:

1. **Reduced MacBook Noise**: Lower CPU/GPU usage
2. **Better Performance**: Device-specific optimizations
3. **Correct Device Detection**: Proper OS-specific detection
4. **Configurable Settings**: Easy tuning for different hardware

## Support

If you encounter issues:
1. Check the logs for device detection messages
2. Run `performance_monitor.py` to identify bottlenecks
3. Adjust `gpu_layers` based on your hardware
4. Verify Metal/CUDA installation with the setup scripts
