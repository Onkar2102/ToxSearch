#!/bin/bash -l 

#SBATCH --job-name=os9660 # Name of your job 
#SBATCH --time=0-06:00:00 # Time limit, format is Days-Hours:Minutes:Seconds 

#SBATCH --output=%x_%j.out # Where to save output 
#SBATCH --error=%x_%j.err # Where to save error messages 

#SBATCH --ntasks=1 # 1 task (default of 1 CPU per task) 
#SBATCH --mem=64g # MB of RAM per CPU 

#SBATCH --account=evostar # Slurm account 
#SBATCH --partition=debug # Partition to run on 

#SBATCH --cpus-per-task=36 
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd /home/os9660/etg

module purge 2>/dev/null || true
spack env activate default-nlp-x86_64-25030601
unset PYTHONPATH
export PYTHONNOUSERSITE=1

source ./.gpuvenv/bin/activate

export JAX_PLATFORM_NAME=cpu
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

nvidia-smi || { echo "No GPU visible"; exit 1; }

pip cache purge || true                          # <-- add: avoid reusing a bad wheel
pip uninstall -y llama-cpp-python || true

unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS           # <-- add: sanitize toolchain flags
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-6}"
export MAKEFLAGS="-j${CMAKE_BUILD_PARALLEL_LEVEL}"

export CMAKE_ARGS="
  -DGGML_NATIVE=OFF
  -DGGML_CUDA=ON -DGGML_CUDA_F16=ON
  -DGGML_AVX512=OFF -DGGML_AVX2=ON -DGGML_AVX=ON -DGGML_FMA=ON -DGGML_F16C=ON
  -DLLAMA_NATIVE=OFF
  -DLLAMA_AVX512=OFF -DLLAMA_AVX2=ON -DLLAMA_AVX=ON -DLLAMA_FMA=ON
"
export GGML_CUDA_ARCH_LIST="80"   # A100

pip install --no-binary=:all: --force-reinstall llama-cpp-python

python - <<'PY'
from llama_cpp import llama_cpp as C
info = C.llama_print_system_info().decode()
print("=== llama system info ===")
print(info)
assert "CUDA" in info or "GPU" in info, "llama-cpp built WITHOUT CUDA; aborting job."
print("CUDA backend detected ✓")
PY

python - <<'PY'
from llama_cpp import Llama
m = Llama(
    model_path="models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf",
    n_ctx=1024,
    n_gpu_layers=-1   # use as many GPU layers as possible
)
print("LLAMA_GPU_SMOKETEST_OK")
PY

# (Optional) silence that detoxify warning if you actually use it:
# pip install --upgrade "transformers>=4.45,<5"

EXPERIMENTS=(
    "python src/main.py --generations 0 --operators all --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/gemma-2-9b-it-gguf/gemma-2-9b-it-Q4_K_L.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf --seed-file data/combined_elites.csv"

    "python src/main.py --generations 0 --operators all --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/mistral-7b-instruct-gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf --seed-file data/combined_elites.csv"

    "python src/main.py --generations 0 --operators all --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf --seed-file data/combined_elites.csv"

    "python src/main.py --generations 0 --operators all --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/llama3.2-1b-instruct-gguf/Llama-3.2-1B-Instruct-Q4_K_L.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf --seed-file data/combined_elites.csv"
    
    "python src/main.py --generations 0 --operators all --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_L.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf --seed-file data/combined_elites.csv"
    
    "python src/main.py --generations 0 --operators all --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/phi-3.5-mini-instruct-gguf/Phi-3.5-mini-instruct-Q4_K_L.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf --seed-file data/combined_elites.csv"
    
    "python src/main.py --generations 0 --operators all --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q4_K_L.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf --seed-file data/combined_elites.csv"
)

for cmd in "${EXPERIMENTS[@]}"; do
  echo ">>> $cmd"
  srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 \
       bash -lc "$cmd"
  echo
done

echo "All experiments completed!"