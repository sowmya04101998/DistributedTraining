
# **Multi-GPU Training**

This guide explains how to train a CNN model on the MNIST dataset using multiple GPUs. It builds upon the single-GPU training approach and introduces `DataParallel` for leveraging multiple GPUs. This approach helps to accelerate training by distributing computation across available GPUs.

## **Step 1: Ensure the Environment is Active**

Ensure that the required Conda environment is already active. If not, activate it:

```bash
$ module load miniconda
$ conda activate gujcost_workshop
```


## **Step 2: Prepare the MNIST Dataset**

If the dataset is not already available, download it by running the following command on the login node:

```bash
(gujcost_workshop) $ python download_data.py
```

Alternatively, if the dataset is already downloaded from a single-GPU workflow, you can copy or move the dataset:

```bash
(gujcost_workshop) $ cp -r /path/to/existing/mnist/data ./data
```



## **Step 3: Inspect the Multi-GPU Training Script**

The script `mnist_multigpu.py` contains the implementation for training using multiple GPUs. Navigate to the directory and view its content:

```bash
(gujcost_workshop) $ cd path/to/multi_gpu_training
(gujcost_workshop) $ cat mnist_multigpu.py
```

### **Key Features of the Script**:
1. **`DataParallel` Usage**:
   Enables the model to utilize multiple GPUs efficiently:
   ```python
   if use_cuda and torch.cuda.device_count() > 1:
       print(f"Using {torch.cuda.device_count()} GPUs.")
       model = nn.DataParallel(model)
   ```

2. **Optimizations**:
   - Experiment with optimizers like `Adam` or `RMSprop` for faster convergence.
   - Implement learning rate schedulers like `StepLR`, `ExponentialLR`, or `CosineAnnealingLR`.


## **Step 4: SLURM Script for Multi-GPU Training**

Create a SLURM script (`slurm_multigpu.sh`) with the following content:

```bash
#!/bin/bash
#SBATCH --job-name=mnist_multiGPU  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=2                 # Number of tasks
#SBATCH --cpus-per-task=10         # Number of CPU cores per task
#SBATCH --partition=gpu            # GPU partition
##SBATCH --reservation=hpcai      # Reservation incase of urgent requirement
##SBATCH --nodelist=rpgpu*        # Specify reservation GPU node name provided
#SBATCH --gres=gpu:2               # Number of GPUs
#SBATCH --output=logs_%j.out       # Output log file
#SBATCH --error=logs_%j.err        # Error log file
#SBATCH --time=00:20:00            # Time limit

# Log the node and GPUs being used
echo "Running on host $(hostname)"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Load required modules
module purge
module load miniconda

# Activate the Conda environment
conda activate gujcost_workshop

# Run the script
kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_multigpu.py --epochs=5 --batch-size=128
```
Changes made are follows:
```bash
#SBATCH --ntasks=2                 # Number of tasks
#SBATCH --gres=gpu:2               # Number of GPUs
```

## **Step 5: Submit the SLURM Job**

Submit the job using the following command:

```bash
(gujcost_workshop) $ sbatch slurm_multigpu.sh
```

Once submitted, the job will run on the allocated GPUs, and the output and error logs will be saved in the specified files.



## **Step 6: Analyze the Profiling Data**

After the job completes, analyze the profiling results:

```bash
(gujcost_workshop) $ python -m line_profiler -rmt <SLURM_JOB_ID>_<SLURM_CPUS_PER_TASK>.lprof
```

Example output:

```
Timer unit: 1e-06 s

Total time: 30.689 s
File: mnist_multigpu.py
Function: train at line 39

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    39                                           @profile
    40                                           def train(args, model, device, train_loader, optimizer, epoch):
    41         5        625.3    125.1      0.0      model.train()
    42      2350    5394001.3   2295.3     17.6      for batch_idx, (data, target) in enumerate(train_loader):
    43      2345     327305.9    139.6      1.1          data, target = data.to(device), target.to(device)
    44      2345     107339.3     45.8      0.3          optimizer.zero_grad()
    45      2345   19431190.0   8286.2     63.3          output = model(data)
    46      2345     153701.3     65.5      0.5          loss = F.nll_loss(output, target)
    47      2345    4366436.8   1862.0     14.2          loss.backward()
    48      2345     844439.3    360.1      2.8          optimizer.step()
    49      2345       2873.9      1.2      0.0          if batch_idx % args.log_interval == 0:
    50       705       4179.2      5.9      0.0              print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
    51       470      56887.4    121.0      0.2                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

```

The profiling results will help identify bottlenecks in the training loop.



## **Step 7: Experimentation Options**

Participants can try the following optimizations:
1. **Experiment with Optimizers**:
   - Replace `Adadelta` with `Adam` or `RMSprop` to observe performance changes.

2. **Change Batch Size**:
   - Increase or decrease the batch size to study the impact on GPU utilization.

3. **Adjust Learning Rate Schedulers**:
   - Use `StepLR`, `ExponentialLR`, or `CosineAnnealingLR` to tune learning rate schedules.


### **Update Slurm Settings**
Change the Slurm script to utilize more CPU cores:

```bash
#SBATCH --cpus-per-task=4
```
```bash
#SBATCH --cpus-per-task=8
```
Submit the job again and observe the performance improvement.

## **Summary**

This guide outlines the process of training a CNN model on the MNIST dataset using multiple GPUs. Profiling and experimentation are key to understanding the performance gains from distributed training. By leveraging multi-GPU setups, you can significantly reduce training time and improve model scalability.


