# Importing required libraries
import argparse  # For handling command-line arguments
import torch  # PyTorch library for tensor operations
import torch.nn as nn  # PyTorch modules for neural network layers
import torch.nn.functional as F  # Functions for activations, losses, etc.
import torch.optim as optim  # PyTorch optimization algorithms
import os  # For accessing environment variables and file handling
import time  # For measuring throughput
from torchvision import datasets, transforms  # For data handling and preprocessing
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training

# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Initialize a GradScaler for mixed precision
scaler = GradScaler()

# Training loop with Mixed Precision & Throughput Measurement
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    start_time = time.time()  # Start time measurement
    total_samples = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Enable mixed precision with autocast
        with autocast():
            output = model(data)
            loss = F.nll_loss(output, target)

        # Scale loss and perform backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_samples += len(data)
        total_loss += loss.item()

        if args.dry_run:
            break

    elapsed_time = time.time() - start_time
    throughput = total_samples / elapsed_time if elapsed_time > 0 else 0
    avg_loss = total_loss / len(train_loader)

    print(f'Train Epoch: {epoch} | Average Loss: {avg_loss:.6f} | Throughput: {throughput:.2f} samples/sec')

# Testing loop with Throughput Measurement
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    start_time = time.time()  # Start time measurement

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)

    elapsed_time = time.time() - start_time
    throughput = total_samples / elapsed_time if elapsed_time > 0 else 0
    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%) | Throughput: {throughput:.2f} samples/sec\n')

# Main function
def main():
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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Set device to CUDA if available and not disabled
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    if use_cuda:
        cuda_kwargs = {'num_workers': num_workers, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
