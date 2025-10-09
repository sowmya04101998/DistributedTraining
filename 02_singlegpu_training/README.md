
# **Single-GPU Training with PyTorch **

Optimizing your script for a **single GPU** is a crucial first step before scaling to multiple GPUs. Inefficient single-GPU code may result in **wasted resources and longer queue times** when running multi-GPU jobs. This guide demonstrates how to train a **CNN on the MNIST dataset** using a **single GPU**.

---

## **Step 1: Data and Model are available on home Directory**
Navigate to the **single-GPU directory**:

Follow the main directory

```bash
$ cd DistributedTraining/02_singlegpu_training
```

---

## **Step 2: Inspect the Training Scripts**

Navigate to the directory and view the content of the training scripts:

```bash
$ ls
01_mnist_model.py  # Standard PyTorch Training
02_mnist_dali.py   # PyTorch + NVIDIA DALI for Data Loading
03_mnist_amp.py    # PyTorch + Automatic Mixed Precision (AMP)
```

Analyze the code to understand different approaches and optimizations.

---

## **Step 3: Understanding the Training Script**

### **Basic Training Script (`01_mnist_model.py`)**
- **Loads the MNIST dataset**
- **Trains a CNN model on a single GPU**
- **Uses PyTorch's native DataLoader for data loading**

#### **Initialize Model, Optimizer, and Scheduler**
```python
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
```

#### **Training Loop**
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
```

---

## **Step 4: Optimized Training with NVIDIA DALI (`02_mnist_DALI.py`)**
[**NVIDIA DALI**](https://docs.nvidia.com/deeplearning/dali/) is a highly optimized data loading and augmentation library that:
- **Loads images directly on the GPU** to reduce CPU bottlenecks.
- **Optimizes performance for large datasets**.
- **Improves training throughput** by parallelizing data preprocessing.

#### **DALI Pipeline for Data Loading**
```python
@pipeline_def
def mnist_pipeline(data_path, batch_size):
    files, labels = fn.readers.file(file_root=data_path, random_shuffle=True, name="Reader")
    images = fn.decoders.image(files, device="mixed", output_type=types.GRAY)
    images = fn.resize(images, resize_x=28, resize_y=28)
    images = fn.crop_mirror_normalize(images, dtype=types.FLOAT, mean=[0.1307], std=[0.3081])
    return images, labels
```

### **Key Differences from Standard PyTorch**
- **DALI loads images on the GPU**, reducing CPU-to-GPU transfer bottlenecks.  
- **Faster training throughput**, especially when dataset size increases.

---

## **Step 5: Training with Automatic Mixed Precision (AMP) (`03_mnist_AMP.py`)**
AMP **reduces GPU memory usage and speeds up training** by:
- **Using FP16 precision** where possible.
- **Scaling loss to prevent underflows**.

#### **Enable AMP in Training Loop**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        with autocast():  # Enables mixed precision
            output = model(data)
            loss = F.nll_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### **Key Differences from Standard PyTorch**
1. **Speeds up training using FP16 precision** where applicable.  
2.  **Uses `autocast()` to automatically switch between FP16 and FP32.**  
3. **Prevents underflow using `GradScaler()`.**  

---

## **Step 6: Submitting the Training Job via SLURM**
### **SLURM Script for Single-GPU Training (`01_slurm_submit.sh`)**
Create a SLURM job script:
```bash
#!/bin/bash
#SBATCH --job-name=mnist_1GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --reservation=SCA
#SBATCH --gres=gpu:1
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err
#SBATCH --time=00:10:00

echo "Running on host:" $(hostname)

module purge
module load miniconda

conda activate tutorial

time python 01_mnist_model.py --epochs=10
```

### **Submit the job**
```bash
$ sbatch 01_slurm_submit.sh
```

### **Monitor GPU Usage**
```bash
$ ssh <gpu_node>
$ watch -n 0.1 nvidia-smi
$ top -u <hpcusername>
```

---

## **Step 7: Analyzing Performance - Training Time and Throughput**
To analyze **training time and throughput**, modify the SLURM job script to run different versions of the code.


### **Run Different Configurations**
#### **1. Run without GPU**
```bash
python 01_mnist_model.py --epochs=5 --no-cuda
```
#### **2. Test with Different Batch Sizes**
```bash
python 01_mnist_model.py --epochs=10 --batch-size=128
python 01_mnist_model.py --epochs=10 --batch-size=256
```
#### **3. Run on Multiple workers**
Modify `01_slurm_submit.sh`:
```bash
#SBATCH --cpus-per-task=1
```
changes number of CPU cores to 4,8 and check the performance of model

#### **3. Run on Multiple GPUs**
Modify `slurm_submit.sh`:
```bash
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
```
Check if **multi-GPU training works**.

---

## **Step 8: Setting Up the Conda Environment**
To recreate the Conda environment:
```bash
$ module load miniconda
$ conda create --name tutorial python=3.9 --yes
$ conda activate tutorial
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
$ pip install --extra-index-url https://pypi.ngc.nvidia.com nvidia-dali-cuda110
$ pip install transformers
$ pip install scikit-learn
```

---

## **Step 9: Results Analysis**
| **Configuration** | **Throughput (images/sec) 8 Threads** | **Total Time (s) - 8 Threads** |
|------------------|------------------------|----------------|
| **Standard PyTorch** (01_mnist_model.py) | ? | ? |
| **DALI Optimized** (02_mnist_dali.py) | ? | ? |
| **AMP Enabled** (03_mnist_amp.py) | ? | ? |

_Note results after running the experiments._

---

## **Summary**
- **Standard PyTorch training is the baseline**.
- **DALI optimizes data loading, improving training speed**.
- **AMP reduces memory usage and accelerates training**.
- **Throughput and total training time should be analyzed**.
- **Efficient single-GPU training is necessary before scaling to multiple GPUs**.

---
## **Next Steps**
Once you've optimized single-GPU training, proceed to **[Multi-GPU Data Parallel Training](../03_multigpu_dp_training/)**.

---
