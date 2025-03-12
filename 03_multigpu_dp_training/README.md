## **Multi-GPU Training with PyTorch**
This guide extends the **single-GPU training setup** to a **multi-GPU environment**, leveraging:
- **PyTorch’s `DataParallel` module** for training across multiple GPUs.
- **Data loading with increased `num_workers`.**

---

## **Step 1: Follow Single-GPU Setup**
Before proceeding, complete the setup in the **[Single-GPU Training Guide](../02_singlegpu_training/)**.

---

## **Step 2: Transition to Multi-GPU**
Navigate to the **multi-GPU directory**:
```bash
$ cd SCA-2025-DistributedTraining/DistributedTraining/03_multigpu_dp_training
```
This directory contains the **modified training scripts** and **Slurm job scripts** optimized for **multi-GPU training**.

---

## **Step 3: Changes from Single-GPU to Multi-GPU Training**
### ** Enable Multi-GPU with `DataParallel`**
Modify the model initialization to **automatically use multiple GPUs**:
```python
# Initialize the CNN model
model = Net()

# Enable multi-GPU support using DataParallel
if use_cuda and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)

model = model.to(device)
```
---

## **Step 4: Slurm Script for Multi-GPU Training**
Create a **Slurm submission script** (`slurm_submit.sh`) to allocate multiple GPUs:

```bash
#!/bin/bash
#SBATCH --job-name=mnist_multiGPU  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=2                 # Number of tasks
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
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
time python mnist_multigpu.py --epochs=10 --batch-size=128
```

Submit the job:
```bash
$ sbatch slurm_submit.sh
```

---

## **Step 5: Monitor GPU and System Performance**
### **Check GPU Utilization**
```bash
$ ssh <gpu_node>
$ watch -n 0.1 nvidia-smi
```
### **Monitor CPU Utilization**
```bash
$ top -u <hpcusername>
```

---

## **Step 6: Throughput Analysis (Multi-GPU Performance)**
Compare performance for **different setups**.

| **Configuration** | **Throughput (images/sec) - 8 Threads** | **Total Time (s) - 8 Threads** |
|------------------|---------------------------------|----------------------|
| **Single GPU (01_mnist_model.py)** | ? | ? |
| **Multi-GPU (mnist_multigpu.py, `DataParallel`)** | ? | ? |

### **Run Experiments**
**Baseline (Single GPU)**
```bash
python 01_mnist_model.py --epochs=10 --batch-size=128
```
**Multi-GPU Training**
```bash
python mnist_multigpu.py --epochs=10 --batch-size=256
```
**Benefit**: **Find the best-performing setup!**

---

## **Summary**
- **Multi-GPU training accelerates deep learning workloads**.
- **Throughput analysis helps measure performance gains**.

---

## **Why Choose `Distributed Data Parallel (DDP)` Over `DataParallel`?**

### **Key Differences**
| Feature | `DataParallel` (DP) | `Distributed Data Parallel` (DDP) |
|----------|--------------------|---------------------------------|
| **Ease of Use** | Simple to implement (`model = nn.DataParallel(model)`) | Requires setting up process groups |
| **Performance** | High inter-GPU communication overhead | Efficient communication, optimized for scalability |
| **Scalability** | Limited to a single node | Works across multiple nodes and GPUs |
| **Gradient Synchronization** | Performed on the main GPU, causing bottlenecks | Each GPU synchronizes gradients independently |
| **Recommended Use** | Small-scale models, quick prototyping | Large-scale training, multi-GPU/multi-node setups |

### **Why `DataParallel` Has Limitations**
- **Replicates the model on every GPU** in each forward pass.
- **Synchronizes gradients on the main GPU**, creating a **bottleneck**.
- **Inefficient scaling beyond a single node**.

### **Why `Distributed Data Parallel (DDP)` Is Preferred**
- **Each GPU has its own dedicated process**, avoiding bottlenecks.
- **Minimizes inter-GPU communication overhead**, improving efficiency.
- **Scales seamlessly across multiple GPUs and nodes** for large-scale training.

### **Conclusion**
If you’re training deep learning models on multiple GPUs, **DDP is the preferred approach** due to its efficiency, scalability, and performance benefits.

[Proceed to Distributed Data Parallel Training](../04_multigpu_ddp_training/)
