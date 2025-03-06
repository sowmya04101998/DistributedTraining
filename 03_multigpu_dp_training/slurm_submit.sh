#!/bin/bash
#SBATCH --job-name=mnist_multiGPU  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=2                 # Number of tasks
#SBATCH --cpus-per-task=6          # Number of CPU cores per task
#SBATCH --partition=gpu            # GPU partition
#SBATCH --reservation=SCA          # Reservation
#SBATCH --gres=gpu:2               # Number of GPUs (adjust as needed)
#SBATCH --output=logs_%j.out       # Output log file
#SBATCH --error=logs_%j.err        # Error log file
#SBATCH --time=00:20:00            # Time limit

# Log the node and GPUs being used
echo "Running on host $(hostname)"

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Load required modules
module purge
module load miniconda

#activate the environment
conda activate tutorial

# Run the script
time python mnist_multigpu.py --epochs=6 --batch-size=128

