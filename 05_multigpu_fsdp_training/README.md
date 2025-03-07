---

# **Fully Sharded Data Parallel (FSDP) Training**

This guide explains how to train a **CodeLlama** model on chess data using **PyTorch's Fully Sharded Data Parallel (FSDP)** framework.  
FSDP **efficiently shards model parameters, gradients, and optimizer states** across GPUs, **reducing memory usage and improving scalability** compared to Data Parallel (DP) and Distributed Data Parallel (DDP).

---

## **Prerequisites**

1. Follow the steps in the **[Single-GPU Training Guide](../02_singlegpu_training/)** to set up your environment and repository.
2. Ensure the `tutorial` Conda environment is activated:

   ```bash
   $ module load miniconda
   $ conda activate tutorial
   ```

3. **Ensure GPU availability** using:
   ```bash
   nvidia-smi
   ```

4. **Follow the Single-GPU guide to download the dataset** before running multi-GPU training.

---

## **Step 1: Key Changes for FSDP**

### **1️⃣ Why Use FSDP Instead of DDP?**
| Feature            | Data Parallel (DP) | Distributed Data Parallel (DDP) | Fully Sharded Data Parallel (FSDP) |
|--------------------|-------------------|---------------------------------|------------------------------------|
| **Memory Usage**   | High              | Medium                          | Low (Parameters, Gradients, and Optimizer are sharded) |
| **Scalability**    | Limited           | Good                            | Excellent (Better for Large Models) |
| **Communication Overhead** | High | Medium | Low (Reduces GPU Memory Communication) |
| **Best For**       | Small Models      | Medium-Sized Models             | Large Models (e.g., CodeLlama, GPT) |

---

## **Step 2: Training Script for FSDP**

The FSDP training script (`fsdp_finetune.py`) includes all necessary configurations.  
Key sections include:

### **1️⃣ Initialize Distributed Training**
```python
torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(torch.distributed.get_rank())
device = torch.cuda.current_device()
```

### **2️⃣ Fully Sharded Data Parallel (FSDP)**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Wrap model with FSDP
model = FSDP(model, device_id=device)
```

### **3️⃣ Optional: Use Layer-Wise Sharding Policy**
```python
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import functools

def layer_policy_fn(module):
    return "layer" in module.__class__.__name__.lower()

auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=layer_policy_fn)

# Apply policy while wrapping model
model = FSDP(model, device_id=device, auto_wrap_policy=auto_wrap_policy)
```

### **4️⃣ Gradient Accumulation & Mixed Precision**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    outputs = model(**batch, use_cache=False)
    loss = outputs.loss / args.gradient_accumulation_steps
```

---

## **Step 3: FSDP Slurm Batch Script**

Create a Slurm submission script (`slurm_submit_fsdp.sh`) to configure and execute the FSDP script:

```bash
#!/bin/bash -l

#SBATCH --job-name=FSDP_finetune  
#SBATCH --output=%x-%j.out             
#SBATCH --error=%x-%j.err              
#SBATCH --nodes=1                      
#SBATCH --ntasks=2                     
#SBATCH --cpus-per-task=10             
#SBATCH --gres=gpu:2                   
#SBATCH --time=04:00:00                
#SBATCH --partition=gpu                
#SBATCH --reservation=SCA              

# Load Environment
module purge
module load miniconda
conda activate tutorial  

# Set Training Parameters
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

total_batch_size=${TOTAL_BATCH_SIZE:-8}
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-4}
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
gradient_accumulation_steps=$(( total_batch_size / batch_size_per_device / num_gpus ))

# Debugging Information
echo "------------------------"
echo "Training Configuration:"
echo "Using $num_gpus GPUs"
echo "Total batch size: $total_batch_size"
echo "Batch size per device: $batch_size_per_device"
echo "Gradient accumulation steps: $gradient_accumulation_steps"
echo "------------------------"

# Launch Distributed Training
torchrun \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    fsdp_finetune.py \
        --batch_size_per_device $batch_size_per_device \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        "$@"
```

Submit the job:
```bash
sbatch slurm_submit_fsdp.sh
```

---

## **Step 4: Monitor and Analyze Performance**

### **Check Logs**
Monitor logs in real-time:
```bash
tail -f FSDP_finetune-<job_id>.out
```

### **Check GPU Utilization**
Monitor GPU utilization:
```bash
nvidia-smi
```

Monitor system resources:
```bash
top -u $USER
```

---

## **Step 5: Performance Considerations**
### **1️⃣ Optimize CPU Utilization**
Ensure **OMP_NUM_THREADS** is correctly set:
```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
```

### **2️⃣ Adjust Batch Sizes and Gradient Accumulation**
| GPUs | Batch Size Per Device | Total Batch Size | Gradient Accumulation Steps |
|------|----------------------|-----------------|--------------------------|
| 2    | 4                    | 8               | 1                        |
| 4    | 8                    | 32              | 2                        |
| 8    | 16                   | 128             | 4                        |

Modify in `slurm_submit_fsdp.sh`:
```bash
total_batch_size=32
batch_size_per_device=8
gradient_accumulation_steps=$(( total_batch_size / batch_size_per_device / num_gpus ))
```

### **3️⃣ Use Gradient Checkpointing**
Enable it via:
```python
model.gradient_checkpointing_enable()
```
Run with:
```bash
fsdp_finetune.py --gradient_checkpointing
```

---

## **Step 6: Assignments**
1. Change `--cpus-per-task` in `slurm_submit_fsdp.sh` to 4 or 8 and analyze performance.
2. Adjust `--batch_size_per_device` to **4, 8, 16** and check training speed.
3. Compare training time between **DDP vs. FSDP** by running:
   ```bash
   fsdp_finetune.py --no_fsdp
   ```
4. Use `gradient_checkpointing` and observe memory savings.

---

## **Summary**
- **FSDP reduces memory usage** and **improves efficiency** for large models.
- **Batch size and gradient accumulation** tuning is crucial.
- **Gradient checkpointing can further optimize memory usage**.
- **FSDP scales better than DDP for large-scale models**.

For more details, visit: [PyTorch FSDP Docs](https://pytorch.org/docs/stable/fsdp.html)

[Go to Multi-GPU DDP Training](../04_multigpu_ddp_training/)

---
