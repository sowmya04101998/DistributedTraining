# **Single-GPU Training**

Optimizing your script for a single GPU is a crucial first step before scaling to multiple GPUs. Inefficient single-GPU code may result in wasted resources and longer queue times when running multi-GPU jobs. This example demonstrates how to train a CNN on the MNIST dataset using a single GPU


---

## **Step 1: Copy Data and Model from common directory**

```bash
$ cp -r /home/apps/SCA-tutorial .
```

## **Step 2: Inspect the training script and python code**

### Inspect the Training Script
Navigate to the directory and view the content of the training script:

```bash
$ cat mnist_model.py
```
Analyze the code for a better understanding of its structure and workflow.


### Script Sections

#### Initialize Model, Optimizer, and Scheduler
```python
# Initialize model, optimizer, and scheduler
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
```

#### Configure DataLoader with CUDA Optimization
```python
if use_cuda:
    cuda_kwargs = {'num_workers': num_workers, 'pin_memory': True, 'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
```

#### Define Dataset Transformations
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize MNIST images
])
```

#### Initialize DataLoaders
```python
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
```

#### Training Function with Profiling
```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' 
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
```

## **Step 4: Slurm Script for Single-GPU Training**

Create a `slurm_submit.sh` file:

```bash
#!/bin/bash
#SBATCH --job-name=mnist_1GPU    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Specify GPU partition for GPU Nodes
#SBATCH --reservation=hpcai      # Reservation incase of urgent requirement
##SBATCH --nodelist=rpgpu*        # Specify reservation GPU node name provided
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=logs_%j.out     # output logfile name
#SBATCH --error=logs_%j.err      # error logfile name
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

# which gpu node was used
echo "Running on host of RUDRA" $(hostname)

module purge #remove unneccesary loaded modules

#load the module
module load miniconda

#activate the environment
conda activate gujcost_workshop

#Try both the option, See the performance

kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_model.py --epochs=5
```

Submit the job:

```bash
(gujcost_workshop) $ sbatch slurm_submit.sh
```

### Monitor GPU and System Performance
Once allocated, monitor the GPU and system threads:
```bash
$ ssh <gpu_node>
$ watch -n 0.1 nvidia-smi
$ top -u <hpcusername>
```
This will help you observe GPU utilization and thread/process spawning during execution.



## **Step 5: Analyze Profiling Data**

After the job completes, analyze the profiling results:

```bash
(gujcost_workshop) $ python -m line_profiler -rmt <job_id>_<slurmtask>.lprof
```

---
---

## **Assignments**

Note: To edit the files use nano editor in linux environment

### use nano editor to edit files

```bash
(gujcost_workshop) $ nano mnist_model.py
```
and make changes as per assignment `ctrl + x` and type `yes` then `Enter`

1. Adjust `--cpus-per-task` to values like 2, 4, 6, 8, or 10 analyze the time and resource utilization (`slurm_submit.sh`).

2. Use the following line to test performance without GPU(`slurm_submit.sh`):
  ```bash
  kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_model.py --epochs=5 --no-cuda
  ```
3. Change the batchsize of dataloader to 128 and 256 see the performance(`slurm_submit.sh`):
  ```bash
  kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_model.py --epochs=5 --batch-size=128
  ```
5. Change `--ntasks=2` and set `--gres=gpu:2` see if code is runnin on multiple GPUs ?

---
---

## **How the Conda Environment Was Created**

```bash
$ module load miniconda
$ conda create --name tutorial python=3.9 --yes
$ conda activate tutorial
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
$ pip install --extra-index-url https://pypi.ngc.nvidia.com nvidia-dali-cuda110clear
```


## **Summary**

Optimizing single-GPU training is a critical step before scaling to multiple GPUs. Efficient code ensures reduced resource usage and shorter queue times. By profiling and addressing bottlenecks, you can achieve higher GPU utilization and better performance.
[Go to Multi-GPU Data Parallel Training](../03_multigpu_dp_training/) for the next steps in distributed training.
