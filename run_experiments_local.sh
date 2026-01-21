#!/bin/bash
# Multiple Experiment Runner
# 
# This script runs multiple experiment executions sequentially.
# Each experiment gets its own output directory: data/outputs/YYYYMMDD_HHMM/
#
# Usage:
#   1. Add your experiments to the EXPERIMENTS array below
#   2. Each experiment is a single string with the full command
#   3. Run: bash run_experiments_local.sh
#
# Features:
#   - Runs experiments one at a time (sequential execution)
#   - 5-second delay between experiments
#   - Shows progress (Experiment X/Total)
#   - Continues even if one experiment fails
#   - Each experiment creates its own timestamped output directory
#
# Tips:
#   - Use comments to label experiments (e.g., "# Experiment 1: Default params")
#   - Vary parameters systematically (theta-sim, theta-merge, generations, etc.)
#   - Each experiment must be a single quoted string (use \ for line continuation)
#   - Make sure all model paths and file paths are correct

set -Eeuo pipefail

# Activate your local virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Define your experiments here (one per line)
# Each experiment runs sequentially with a 5-second delay between them
# Add as many experiments as you want - they will all run automatically
EXPERIMENTS=(
    # Experiment 1: Default parameters
    "python src/main.py \
        --generations 25 \
        --threshold 0.99 \
        --moderation-methods google \
        --stagnation-limit 3 \
        --theta-sim 0.25 \
        --theta-merge 0.40 \
        --species-capacity 5 \
        --cluster0-max-capacity 1000 \
        --cluster0-min-cluster-size 2 \
        --min-island-size 3 \
        --species-stagnation 1 \
        --embedding-model all-MiniLM-L6-v2 \
        --embedding-dim 384 \
        --embedding-batch-size 64 \
        --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf \
        --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf \
        --operators all \
        --max-variants 1 \
        --seed-file data/prompt.csv"

    # qwen 5
  "python src/main.py --generations 120 --operators all --max-variants 1 \
   --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 \
   --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
   --pg models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q4_K_L.gguf \
   --seed-file data/prompt.csv"

       # llama 5
  "python src/main.py --generations 120 --operators all --max-variants 1 \
   --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 \
   --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
   --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
   --seed-file data/prompt.csv"

)

echo "Starting ${#EXPERIMENTS[@]} experiments..."
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    NUM=$((i+1))
    TOTAL=${#EXPERIMENTS[@]}
    
    echo "=========================================="
    echo "Experiment $NUM/$TOTAL"
    echo "=========================================="
    echo "Command: ${EXPERIMENTS[$i]}"
    echo ""
    
    bash -lc "${EXPERIMENTS[$i]}"
    
    if [ $? -eq 0 ]; then
        echo "Experiment $NUM completed successfully"
    else
        echo "Experiment $NUM failed"
    fi
    
    echo ""
    echo "Waiting 5 seconds..."
    sleep 5
    echo ""
done

echo "All experiments completed!"

