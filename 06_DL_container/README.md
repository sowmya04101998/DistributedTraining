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

Once downloaded, extract and create the container:

```bash
enroot create -n pytorch_container nvidia+pytorch+24.11-py3.sqsh
```

You can now run the container using **interactive mode** or **batch mode** via SLURM.

---

## **Step 2: Running Enroot Containers on HPC Nodes**

### **A. Running in Interactive Mode (For Debugging)**
Since login nodes do not have GPUs, first request a **GPU node interactively**:

```bash
srun --gres=gpu:1 --partition=gpu --time=00:30:00 --pty /bin/bash -i
```

Once on a GPU node, start an **interactive container session**:

```bash
enroot start --mount $HOME --rw pytorch_container
```

Now, **run PyTorch scripts inside the container**:

```bash
cd SCA-2025-DistributedTraining/02_singlegpu_training/
python mnist_model.py --epochs=5
```

To verify GPU availability inside the container:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

### **B. Running in Batch Mode with SLURM**
For longer training runs, use **SLURM batch jobs**.

#### **1️⃣ SLURM Batch Script (`pytorch_enroot.sh`)**
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
          --container-mounts=$HOME/SCA-2025-DistributedTraining/02_singlegpu_training/:/workspace \
          sh -c 'cd /workspace && python mnist_model.py --epochs=5'
```

#### **2️⃣ Submit the SLURM Job**
```bash
sbatch pytorch_enroot.sh
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
          --container-mounts=$HOME/SCA-2025-DistributedTraining/03_multigpu_dp_training/:/workspace \
          sh -c 'cd /workspace && python mnist_multigpu.py --epochs=5'
```

Submit the job:

```bash
sbatch pytorch_multiGPU.sh
```

---

## **Step 4: Monitoring and Debugging**

### **1️⃣ Check Running Jobs**
```bash
squeue -u $USER
```

### **2️⃣ Monitor GPU Utilization**
```bash
watch -n 1 nvidia-smi
```

### **3️⃣ Check SLURM Logs**
```bash
tail -f pytorch_job-<job_id>.out
```

---

## **Step 5: Performance Optimization**
### **1️⃣ Adjusting CPU and GPU Settings**
Use **OMP_NUM_THREADS** for optimal CPU performance:

```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
```

### **2️⃣ Enabling Mixed Precision (AMP)**
Modify your PyTorch script to **enable AMP for faster training**:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### **3️⃣ Using Larger Batch Sizes with Gradient Accumulation**
For memory efficiency, adjust:

```bash
torchrun --nproc_per_node=2 mnist_model.py --batch_size_per_device 32 --gradient_accumulation_steps 4
```

---

## **Step 6: Assignments**
1. **Change `--gres=gpu:1` to `--gres=gpu:2`** and analyze training time.
2. **Modify `--batch-size` to 128 or 256** and compare performance.
3. **Enable gradient accumulation (`--gradient_accumulation_steps=4`)** and observe memory efficiency.
4. **Use `torchrun` with `nproc_per_node=2`** for distributed training.

---

## **Summary**
**Enroot provides an efficient, containerized PyTorch environment** for HPC systems.  
**Supports GPU acceleration, multi-node execution, and SLURM integration**.  
**Performance optimizations like AMP and gradient accumulation can significantly improve efficiency**.  

For more details, visit:
- [NGC PyTorch Containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
- [Enroot Documentation](https://github.com/NVIDIA/enroot)
