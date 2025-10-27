#!/bin/bash

# Define your experiments here (one per line)
EXPERIMENTS=(
    "python src/main.py --generations 50 --operators ie --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf"
    "python src/main.py --generations 50 --operators ie --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf"
)

echo "Starting $(echo "$EXPERIMENTS" | wc -l) experiments..."
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    NUM=$((i+1))
    TOTAL=${#EXPERIMENTS[@]}
    
    echo "=========================================="
    echo "Experiment $NUM/$TOTAL"
    echo "=========================================="
    echo "Command: ${EXPERIMENTS[$i]}"
    echo ""
    
    eval "${EXPERIMENTS[$i]}"
    
    if [ $? -eq 0 ]; then
        echo "✓ Experiment $NUM completed"
    else
        echo "✗ Experiment $NUM failed"
    fi
    
    echo ""
    echo "Waiting 5 seconds..."
    sleep 5
    echo ""
done

echo "All experiments completed!"

