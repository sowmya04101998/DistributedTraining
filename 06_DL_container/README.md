# **Enroot and NGC PyTorch Containers on HPC**

## **Introduction**
**Enroot** is a lightweight container runtime optimized for **High-Performance Computing (HPC) environments**. It allows users to efficiently run **NVIDIA GPU Cloud (NGC) containers** and other OCI-compliant images on **SLURM-based clusters**.

This guide provides step-by-step instructions to **import, configure, and run PyTorch containers** on an HPC system using Enroot.

---

## **Step 1: Import and Convert NGC PyTorch Containers**

### **1. Download the PyTorch Container from NGC**
Pull the latest **PyTorch 24.11** container from **NVIDIA NGC**:

```bash
enroot import docker://nvcr.io#nvidia/pytorch:24.11-py3
```

You can now run the container using **interactive mode** or **batch mode** via SLURM.

---

## **Step 2: Running Enroot Containers on HPC Nodes**

### **A. Running in Interactive Mode (For Debugging)**
Since login nodes do not have GPUs, first request a **GPU node interactively**:

```bash
srun --gres=gpu:1 --partition=gpu --time=00:30:00 --pty /bin/bash -i
```

extract and create the container:

```bash
enroot create -n pytorch_container nvidia+pytorch+24.11-py3.sqsh
```
Once on a GPU node, start an **interactive container session**:

```bash
enroot start --mount $HOME --rw pytorch_container
```

Now, **run PyTorch scripts inside the container**:

```bash
cd DistributedTraining/06_DL_container/
python 01_mnist_model.py --epochs=5
```

---

### **B. Running in Batch Mode with SLURM**
For longer training runs, use **SLURM batch jobs**.

#### ** SLURM Batch Script (`01_mnist_1gpu.sh`)**
```bash
#!/bin/bash
#SBATCH --job-name=pytorch_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --reservation=SCA

time srun --container-image=/home/apps/enroot/nvidia+pytorch+24.11-py3.sqsh \
          --container-name=pytorch \
          --container-mounts=$HOME/SCA-2025-DistributedTraining/06_DL_container/:/workspace \
          sh -c 'cd /workspace && python 01_mnist_model.py --epochs=10'
```

#### ** Submit the SLURM Job**
```bash
sbatch 01_mnist_model.sh
```

---

## **Step 3: Running Multi-GPU PyTorch Workloads**
For multi-GPU training, modify the SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=pytorch_multiGPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --reservation=SCA

time srun --container-image=/home/apps/enroot/nvidia+pytorch+24.11-py3.sqsh \
          --container-name=pytorch_multi \
          --container-mounts=$HOME/SCA-2025-DistributedTraining/06_DL_container/:/workspace \
          sh -c 'cd /workspace && python 03_mnist_ddp.py --epochs=10'
```

Submit the job:

```bash
sbatch 03_ngc_ddp_multinode.sh
```
```

---

## **Summary**
**Enroot provides an efficient, containerized PyTorch environment** for HPC systems.  
**Supports GPU acceleration, multi-node execution, and SLURM integration**.  
**Performance optimizations like AMP and gradient accumulation can significantly improve efficiency**.  

For more details, visit:
- [NGC PyTorch Containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
- [Enroot Documentation](https://github.com/NVIDIA/enroot)
