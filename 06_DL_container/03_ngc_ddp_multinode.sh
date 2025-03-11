#!/bin/bash
#SBATCH --job-name=mnist_multi     # Job name
#SBATCH --nodes=2                  # Number of nodes
#SBATCH --ntasks-per-node=1      # Number of tasks (one per GPU per node)
#SBATCH --gres=gpu:2               # Number of GPUs on each node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --partition=gpu            # GPU partition
#SBATCH --output=logs_%j.out       # Output log file
#SBATCH --error=logs_%j.err        # Error log file
#SBATCH --time=00:20:00            # Time limit
#SBATCH --reservation=SCA


# Define variables for distributed setup
nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node IP: $head_node_ip"
# Set environment variables for PyTorch distributed training
export MASTER_ADDR=$head_node_ip   # Set the master node IP address
export MASTER_PORT=29500           # Any available port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))
export RANK=$SLURM_PROCID          # Rank of the current process
export LOGLEVEL=INFO               # Log level for debugging
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Log environment variables for debugging
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"


time srun --container-image=/home/apps/enroot/nvidia+pytorch+24.11-py3.sqsh \
        --container-name=pytorch \
--container-mounts=$HOME/SCA-2025-DistributedTraining/06_DL_container/:/workspace,/var/share/slurm/slurm.taskprolog:/var/share/slurm/slurm.taskprolog \
        sh -c 'cd /workspace && torchrun \
    --nnodes=2 \
    --nproc-per-node=2 \
    --rdzv-id=10 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:29500 \
    03_mnist_ddp.py --epochs=10 --batch-size=128'
