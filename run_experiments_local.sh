#!/bin/bash
set -Eeuo pipefail

# Activate your local virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Define your experiments here (one per line)
EXPERIMENTS=(
    "python src/main.py \
        --generations 11 \
        --threshold 0.99 \
        --moderation-methods google \
        --stagnation-limit 5 \
        --theta-sim 0.4 \
        --theta-merge 0.2 \
        --species-capacity 100 \
        --cluster0-max-capacity 1000 \
        --cluster0-min-cluster-size 2 \
        --min-island-size 2 \
        --max-stagnation 20 \
        --embedding-model all-MiniLM-L6-v2 \
        --embedding-dim 384 \
        --embedding-batch-size 64 \
        --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf \
        --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf \
        --operators all \
        --max-variants 1 \
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

