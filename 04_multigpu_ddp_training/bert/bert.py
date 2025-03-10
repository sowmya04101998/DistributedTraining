import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, BertTokenizerFast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import re
from sklearn.model_selection import train_test_split
import torchvision
from tqdm import tqdm

torchvision.disable_beta_transforms_warning()

# File paths for dataset and token file
TEXT_FILE_PATH = '/home/samritm/GUJCOST_workshop/bert/200k_lines_2.txt'
TOKEN_FILE_PATH = '/home/samritm/GUJCOST_workshop/bert/perfectly_cleaned_words_3.txt'

# Custom dataset for tokenized text data
class BookDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, index):
        return {key: val[index].clone().detach() for key, val in self.encodings.items()}

# Add new tokens to tokenizer
def add_tokens(file_path):
    tokenizer = BertTokenizerFast.from_pretrained('/home/samritm/GUJCOST_workshop/bert/biomedbert')
    with open(file_path, 'r') as file:
        new_tokens = [line.strip() for line in file]
    num_added_toks = tokenizer.add_tokens(new_tokens)
    print(f"[INFO] Added {num_added_toks} new tokens to the tokenizer.")
    return tokenizer

# Prepare tokenized dataset
def prepare_dataset(file_path):
    tokenizer = add_tokens(TOKEN_FILE_PATH)
    with open(file_path, 'r') as f:
        texts = [re.sub(r'[^a-zA-Z, ]+', " ", line.strip()) for line in f.read().split('.') if len(line.split()) >= 20]

    print(f"[INFO] Total valid text lines: {len(texts)}")

    train_texts, test_texts = train_test_split(texts, test_size=0.2)
    print(f"[INFO] Training set size: {len(train_texts)}, Test set size: {len(test_texts)}")

    train_inputs = tokenizer(train_texts, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    train_inputs['labels'] = train_inputs['input_ids'].detach().clone()
    train_dataset = BookDataset(train_inputs)
    return train_dataset

# Distributed training setup
def ddp_setup():
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

# Trainer class for managing model training
class Trainer:
    def __init__(self, model, train_data, optimizer, save_every, snapshot_path):
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            self._load_snapshot()
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self):
        checkpoint = torch.load(self.snapshot_path, map_location=f'cuda:{self.local_rank}')
        self.model.load_state_dict(checkpoint['MODEL_STATE'])
        self.epochs_run = checkpoint['EPOCHS_RUN']

    def train_epoch(self, epoch):
        print(f"[INFO] Starting Epoch {epoch}")
        gpu_data_count = 0  # Counter for samples processed by this GPU

        self.train_data.sampler.set_epoch(epoch)
        self.model.train()
        train_loop = tqdm(self.train_data, desc=f"Epoch {epoch}", leave=True)

        total_loss = 0
        for batch_idx, batch in enumerate(train_loop):
            gpu_data_count += len(batch["input_ids"])  # Count samples on this GPU

            inputs = {key: val.to(self.local_rank) for key, val in batch.items()}
            self.optimizer.zero_grad()
            output = self.model(**inputs)
            loss = output.loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar with GPU and loss information
            train_loop.set_postfix({
                "loss": loss.item(),
                "gpu": self.local_rank,
                "samples_gpu": gpu_data_count
            })

        avg_train_loss = total_loss / len(self.train_data)
        print(f"[INFO] Epoch {epoch} completed with Average Loss: {avg_train_loss:.4f}")
        print(f"[INFO] GPU {self.local_rank} processed {gpu_data_count} samples in Epoch {epoch}")

    def save_snapshot(self, epoch):
        torch.save({
            'MODEL_STATE': self.model.module.state_dict(),
            'EPOCHS_RUN': epoch
        }, self.snapshot_path)
        print(f"[INFO] Snapshot saved at Epoch {epoch} to {self.snapshot_path}")

    def train(self, max_epochs):
        for epoch in range(self.epochs_run, max_epochs):
            self.train_epoch(epoch)
            if epoch % self.save_every == 0 and self.local_rank == 0:
                self.save_snapshot(epoch)

# Prepare DataLoader
def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.distributed.DistributedSampler(dataset))

# Main function
def main(save_every, total_epochs, batch_size, snapshot_path='/home/samritm/GUJCOST_workshop/bert/snapshot.pt'):
    print("[INFO] Initializing Distributed Training...")
    ddp_setup()
    dataset = prepare_dataset(TEXT_FILE_PATH)
    print(f"[INFO] Loaded dataset with {len(dataset)} samples.")
    tokenizer = add_tokens(TOKEN_FILE_PATH)
    model = AutoModelForMaskedLM.from_pretrained('/home/samritm/GUJCOST_workshop/bert/biomedbert')
    model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    print("[INFO] Starting Training Process...")
    trainer.train(total_epochs)
    print("[INFO] Training Complete. Cleaning up...")
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('total_epochs', type=int, help='Total epochs')
    parser.add_argument('save_every', type=int, help='Save frequency')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default: 32)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)
