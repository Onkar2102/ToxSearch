#!/bin/bash -l 

#SBATCH --job-name=os9660 # Name of your job 
#SBATCH --time=0-00:60:00 # Time limit, format is Days-Hours:Minutes:Seconds 

#SBATCH --output=%x_%j.out # Where to save output 
#SBATCH --error=%x_%j.err # Where to save error messages 

#SBATCH --ntasks=1 # 1 task (default of 1 CPU per task) 
#SBATCH --mem=0 # MB of RAM per CPU 

#SBATCH --account=evostar # Slurm account 
#SBATCH --partition=debug # Partition to run on 

#SBATCH --cpus-per-task=36 
## SBATCH --gres=gpu:a100:1

# Load your spack env here 
spack env activate default-nlp-x86_64-25030601


# Load your software here 
source /home/os9660/etg/.gpuvenv/bin/activate

# Your code here 

# Define your experiments here (one per line)
EXPERIMENTS=(
    "python src/main.py --generations 50 --operators ie --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf"
    "python src/main.py --generations 50 --operators cm --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf"
    "python src/main.py --generations 50 --operators all --max-variants 1 --elites-threshold 30 --removal-threshold 3 --stagnation-limit 5 --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_M.gguf"
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
