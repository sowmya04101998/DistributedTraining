import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

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

# Training loop with throughput measurement
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    start_time = time.time()  # Start time measurement
    total_samples = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_samples += len(data)
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        if args.dry_run:
            break

    elapsed_time = time.time() - start_time  # Compute total time
    throughput = total_samples / elapsed_time if elapsed_time > 0 else 0
    avg_loss = total_loss / len(train_loader)

    print(f'Train Epoch: {epoch} | Average Loss: {avg_loss:.6f} | Throughput: {throughput:.2f} samples/sec')

# Testing loop with throughput measurement
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
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='logging interval')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker threads for DataLoader')
    parser.add_argument('--dry-run', action='store_true', default=False, help='Run a single batch for debugging')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    num_workers = int(os.getenv('SLURM_CPUS_PER_TASK', args.num_workers))

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

    print(f"Train set size: {len(dataset1)}")
    print(f"Test set size: {len(dataset2)}")

    model = Net()

    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    model = model.to(device)
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

