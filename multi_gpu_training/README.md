
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
Analyse the code for better understanding
```
# Importing required libraries
import argparse  # For parsing command-line arguments
import torch  # Core PyTorch library
import torch.nn as nn  # For building neural network layers
import torch.nn.functional as F  # For activation functions and losses
import torch.optim as optim  # Optimization algorithms
import os  # For handling environment variables and file paths
from torchvision import datasets, transforms  # For downloading and transforming the MNIST dataset
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler

# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Convolutional layer with 32 filters, kernel size 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # Convolutional layer with 64 filters, kernel size 3x3
        self.dropout1 = nn.Dropout(0.25)  # Dropout to prevent overfitting
        self.dropout2 = nn.Dropout(0.5)  # Dropout before fully connected layers
        self.fc1 = nn.Linear(9216, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes (digits 0-9)

    def forward(self, x):
        x = self.conv1(x)  # Apply first convolution
        x = F.relu(x)  # Apply ReLU activation
        x = self.conv2(x)  # Apply second convolution
        x = F.relu(x)  # Apply ReLU activation
        x = F.max_pool2d(x, 2)  # Apply max pooling to reduce spatial dimensions
        x = self.dropout1(x)  # Apply dropout
        x = torch.flatten(x, 1)  # Flatten tensor for fully connected layers
        x = self.fc1(x)  # Fully connected layer
        x = F.relu(x)  # Apply ReLU activation
        x = self.dropout2(x)  # Apply dropout
        x = self.fc2(x)  # Output layer
        output = F.log_softmax(x, dim=1)  # Apply log softmax for classification
        return output

# Training loop
@profile  # Profiling decorator for performance analysis
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data to the specified device (CPU/GPU)
        optimizer.zero_grad()  # Reset gradients
        output = model(data)  # Forward pass
        loss = F.nll_loss(output, target)  # Calculate negative log-likelihood loss
        loss.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update model parameters
        if batch_idx % args.log_interval == 0:  # Log progress every few batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing loop
def test(model, device, test_loader):
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient computation for inference
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data to the device
            output = model(data)  # Forward pass
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up the batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= len(test_loader.dataset)  # Average loss over the test set
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Main function
def main():
    # Argument parser for configuring training parameters via command-line
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()  # Check if CUDA (GPU) is available
    device = torch.device("cuda" if use_cuda else "cpu")  # Use GPU if available, otherwise CPU

    # Training and testing configurations
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': True}  # Enable efficient data loading
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset
    ])

    # Load MNIST dataset
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    print(f"Train set size: {len(dataset1)}")
    print(f"Test set size: {len(dataset2)}")

    model = Net()  # Instantiate the model

    # Enable multi-GPU support if more than one GPU is available
    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    model = model.to(device)  # Move model to the appropriate device

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # Adadelta optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # Reduce learning rate after every epoch

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)  # Train the model
        test(model, device, test_loader)  # Test the model
        scheduler.step()  # Update learning rate

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")  # Save the trained model

if __name__ == '__main__':
    main()
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


