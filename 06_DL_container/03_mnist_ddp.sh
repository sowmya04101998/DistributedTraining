import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
        return F.log_softmax(x, dim=1)

# Setup DDP
def setup():
    dist.init_process_group("nccl", init_method="env://")

def cleanup():
    dist.destroy_process_group()

# ✅ Simplified Training Function
def train(args, model, train_loader, optimizer, epoch, rank):
    model.train()
    start_time = time.time()
    total_loss, total_samples = 0, 0

    for data, target in train_loader:
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)

    elapsed_time = time.time() - start_time
    avg_loss = total_loss / total_samples
    throughput = total_samples / elapsed_time

    if rank == 0:  # ✅ Ensure only rank 0 prints the log
        print(f"Train Epoch: {epoch} | Average Loss: {avg_loss:.6f} | Throughput: {throughput:.2f} samples/sec")

# ✅ Simplified Testing Function
def test(model, test_loader, rank):
    model.eval()
    start_time = time.time()
    test_loss, correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

    elapsed_time = time.time() - start_time
    avg_loss = test_loss / total_samples
    accuracy = 100. * correct / total_samples
    throughput = total_samples / elapsed_time

    if rank == 0:  # ✅ Ensure only rank 0 prints the log
        print(f"Test set: Average Loss: {avg_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%) "
              f"| Throughput: {throughput:.2f} samples/sec")

# Main function
def main():
    parser = argparse.ArgumentParser(description="PyTorch DDP MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, help="Global batch size for training")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="Input batch size for testing")
    parser.add_argument("--epochs", type=int, default=14, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate step gamma")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model")
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    setup()
    torch.cuda.set_device(rank)

    world_size = dist.get_world_size()
    per_gpu_batch_size = args.batch_size // world_size

    if rank == 0:
        print(f"[INFO] Global batch size: {args.batch_size}, Per-GPU batch size: {per_gpu_batch_size}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=per_gpu_batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler)

    model = Net().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, ddp_model, train_loader, optimizer, epoch, rank)
        test(ddp_model, test_loader, rank)
        scheduler.step()

    if rank == 0 and args.save_model:
        torch.save(ddp_model.state_dict(), "mnist_ddp_model.pth")

    cleanup()

if __name__ == "__main__":
    main()
