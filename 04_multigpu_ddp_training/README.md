# **Distributed Data Parallel (DDP) Training**

This guide extends the single-GPU and multi-GPU Data Parallel (DP) training setup to a **multi-GPU Distributed Data Parallel (DDP)** environment. DDP is highly efficient for large-scale training as it reduces inter-GPU communication overhead and improves training speed compared to DP.

---

## **Step 1: Transition to Distributed Data Parallel (DDP)**

Navigate to the DDP directory:

```bash
$ cd SCA-2025-DistributedTraining/DistributedTraining/04_multigpu_ddp_training
```

This directory contains the modified script and configurations for multi-GPU **DDP** training.

---

## **Step 2: Key Adjustments from DP to DDP**

### Process Group Initialization
DDP requires initializing a process group:

```python
def setup():
    """Initialize the process group for DDP."""
    dist.init_process_group("nccl", init_method="env://")

def cleanup():
    """Destroy the process group after training."""
    dist.destroy_process_group()
```

### Distributed Data Sampler
To ensure data is evenly split across GPUs, use `DistributedSampler`:

```python
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=dist.get_world_size(), rank=rank
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
```

### Wrapping Model with DDP
DDP requires wrapping the model to enable distributed training:

```python
model = Net().to(rank)
ddp_model = DDP(model, device_ids=[rank])
```

---

## **Step 3: DDP Training Script**

The training script (`mnist_ddpmodel.py`) includes all necessary configurations for multi-GPU **DDP** training. 

### Training Function with Throughput Logging
```python
def train(args, model, train_loader, optimizer, epoch, rank):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and rank == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}")
```

---

## **Step 4: DDP Slurm Script for Multi-GPU Training**

Create a Slurm submission script (`slurm_submit_ddp.sh`) to allocate multiple GPUs:

```bash
#!/bin/bash
#SBATCH --job-name=mnist_ddp           # Job name
#SBATCH --nodes=2                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks (one per GPU per node)
#SBATCH --gres=gpu:2                   # Number of GPUs on each node
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --partition=gpu                # GPU partition
#SBATCH --output=logs_%j.out           # Output log file
#SBATCH --error=logs_%j.err            # Error log file
#SBATCH --time=00:20:00                # Time limit
#SBATCH --reservation=SCA

# Define variables for distributed setup
nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Set environment variables for PyTorch distributed training
export MASTER_ADDR=$head_node_ip   # Set the master node IP address
export MASTER_PORT=29900           # Any available port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))
export RANK=$SLURM_PROCID          # Rank of the current process
export LOGLEVEL=INFO               # Log level for debugging
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Log environment variables for debugging
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

# Load required modules and activate Conda environment
module purge

module load miniconda

conda activate tutorial

# Run the PyTorch script with torchrun
time srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=2 \
    --rdzv-id=10 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    mnist_ddpmodel.py --epochs=5 --batch-size=128
```

Submit the job:
```bash
$ sbatch slurm_submit.sh
```

---

## **Step 5: Monitor and Analyze Performance**

Monitor GPU utilization and system performance:
```bash
$ watch -n 0.1 nvidia-smi
$ top -u <hpcusername>
```

Analyze training logs:
```bash
tail -f logs_<job_id>.out
```

---

## **Why Move to Fully Sharded Data Parallel (FSDP)?**

1. **Better Memory Efficiency:** FSDP shards both model parameters and optimizer states, reducing memory usage.
2. **Improved Scalability:** FSDP scales better across multi-node setups for large models.
3. **Gradient Sharding:** Unlike DDP, FSDP distributes gradients, improving performance for massive models.

These advantages make **FSDP** ideal for large-scale distributed training beyond DDP.

[Learn more about DDP on the official PyTorch website](https://pytorch.org/docs/stable/notes/ddp.html)

[Proceed to Fully Sharded Data Parallel (FSDP) Training](../05_multigpu_fsdp_training/)

