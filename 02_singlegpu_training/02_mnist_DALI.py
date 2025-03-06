import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from torch.optim.lr_scheduler import StepLR
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# Define global constants for DALI
N = 1  # Number of GPUs
BATCH_SIZE = 128
IMAGE_SIZE = 28
DATASET_PATH = "/home/apps/SCA-tutorial/DistributedTraining/02_singlegpu_training/mnist_images"

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

# Define DALI pipeline for MNIST dataset
@pipeline_def
def mnist_pipeline(data_path):
    files, labels = fn.readers.file(file_root=data_path, random_shuffle=True, name="Reader")
    images = fn.decoders.image(files, device="mixed", output_type=types.GRAY)
    images = fn.resize(images, resize_x=IMAGE_SIZE, resize_y=IMAGE_SIZE)
    images = fn.crop_mirror_normalize(images, dtype=types.FLOAT, mean=[0.1307], std=[0.3081])
    return images, labels

# Training function with throughput measurement
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    start_time = time.time()  # Start time measurement
    total_samples = 0
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):
        for d in data:
            images, labels = d["data"].to(device), d["label"].long().squeeze().to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()

            total_samples += len(images)
            total_loss += loss.item()

        if args.dry_run:
            break

    elapsed_time = time.time() - start_time  # Compute total time
    throughput = total_samples / elapsed_time if elapsed_time > 0 else 0
    avg_loss = total_loss / len(train_loader)

    print(f'Train Epoch: {epoch} | Average Loss: {avg_loss:.6f} | Throughput: {throughput:.2f} samples/sec')

# Testing function with throughput measurement
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    start_time = time.time()  # Start time measurement

    with torch.no_grad():
        for data in test_loader:
            for d in data:
                images, labels = d["data"].to(device), d["label"].long().squeeze().to(device)
                output = model(images)
                test_loss += F.nll_loss(output, labels, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += len(images)

    elapsed_time = time.time() - start_time
    throughput = total_samples / elapsed_time if elapsed_time > 0 else 0
    test_loss /= len(test_loader) * BATCH_SIZE

    print(f'\nTest set: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader) * BATCH_SIZE} '
          f'({100. * correct / (len(test_loader) * BATCH_SIZE):.0f}%) | Throughput: {throughput:.2f} samples/sec\n')

# Main function
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example with DALI')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='logging interval')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--save-model', action='store_true', default=False, help='Save the model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)

    train_pipe = mnist_pipeline(batch_size=BATCH_SIZE, num_threads=8, device_id=0, data_path=os.path.join(DATASET_PATH, 'train'))
    test_pipe = mnist_pipeline(batch_size=BATCH_SIZE, num_threads=8, device_id=0, data_path=os.path.join(DATASET_PATH, 'test'))

    train_pipe.build()
    test_pipe.build()

    train_loader = DALIGenericIterator(train_pipe, ["data", "label"], reader_name="Reader")
    test_loader = DALIGenericIterator(test_pipe, ["data", "label"], reader_name="Reader")

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_dali_cnn.pt")

if __name__ == '__main__':
    main()
