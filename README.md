

# **Single-GPU Training**

Optimizing your script for a single GPU is a crucial first step before scaling to multiple GPUs. Inefficient single-GPU code may result in wasted resources and longer queue times when running multi-GPU jobs. This example demonstrates how to train a CNN on the MNIST dataset using a single GPU and profile the training process for performance improvements.

```bash
$ ssh <hpcuser>@14.139.62.247 -p 4422
$ git clone [<repo>](https://github.com/kishoryd/DistributedTraining.git)
```

## **Step 1: Activate Conda Environment**

To simplify the setup, use a pre-configured Conda environment. Follow these steps:

```bash
$ module load miniconda
$ conda activate gujcost_workshop
```
## **Step 2: Dowload the MNIST dataset**

Ensure the dataset is downloaded on the login node since compute nodes typically lack internet access:

```bash
(gujcost_workshop) $ python download_data.py
```

## **Step 3: Run and Profile the Script**

First, inspect the training script (`mnist_model.py`) by navigating to its directory and viewing its content:

```bash
(gujcost_workshop) $ cd path/to/single_gpu_training
(gujcost_workshop) $ cat mnist_model.py
```

### **Decorate the Training Function for Profiling**
Add a `line_profiler` decorator to the `train` function:

```python
@profile
def train(args, model, device, train_loader, optimizer, epoch):
    ...
```

### **Slurm Script for Single-GPU Training**

Create a `slurm_submit.sh` file with the following content:

```bash
#!/bin/bash
#SBATCH --job-name=mnist_1GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err
#SBATCH --time=00:10:00

echo "Running on host: $(hostname)"

module purge
module load miniconda
conda activate gujcost_workshop

kernprof -o ${SLURM_JOBID}.lprof -l mnist_model.py --epochs=5
```

## **Step 4: Submit job to GPU Node**

```bash
(gujcost_workshop) $ sbatch slurm_submit.sh
```

The job will execute, and the results will be logged.


## **Step 5: Analyze Profiling Data**

After the job completes, analyze the profiling results:

```bash
(gujcost_workshop) $ python -m line_profiler -rmt logs_<job_id>.lprof
```

Example output:

```
Timer unit: 1e-06 s

Total time: 93.0049 s
File: mnist_model.py
Function: train at line 39

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    39                                           @profile
    40                                           def train(args, model, device, train_loader, optimizer, epoch):
    ...
```

The profiling data highlights bottlenecks in the training loop, such as data transfer or model computation.


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

Optimizing single-GPU training is a critical step before scaling to multiple GPUs. Efficient code ensures reduced resource usage and shorter queue times. By profiling and addressing bottlenecks, you can achieve higher GPU utilization and better performance.


## **How the Conda Environment Was Created**
If needed, create the Conda environment with these commands:

```bash
$ module load miniconda
$ conda create --name gujcost_workshop python=3.9 --yes
$ conda activate gujcost_workshop
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
$ conda install line_profiler --channel conda-forge
```

