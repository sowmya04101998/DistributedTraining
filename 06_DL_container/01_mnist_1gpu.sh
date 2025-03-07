#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --exclusive
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --reservation=SCA

time srun --container-image=/home/apps/enroot/nvidia+pytorch+24.11-py3.sqsh \
        --container-name=pytorch \
--container-mounts=$HOME/SCA-2025-DistributedTraining/02_singlegpu_training/:/workspace,/var/share/slurm/slurm.taskprolog:/var/share/slurm/slurm.taskprolog \
        sh -c 'cd /workspace && python 01_mnist_model.py --epochs=5'
