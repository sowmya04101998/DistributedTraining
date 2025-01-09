# **Deep Learning Training Examples with PyTorch**

This repository provides a comprehensive guide and practical examples for training deep learning models using PyTorch across various parallelism strategies. Whether you are working on single-GPU training or scaling to multi-GPU setups with Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP), these examples will guide you through the process.

---

## **Contents**

### 01. **Introduction to Deep Learning**
- Foundational concepts of deep learning and PyTorch.
- Basics of tensors, datasets, and model building.

[Read the Guide](../01_introduction_to_deeplearning/)

---

### 02. **Single-GPU Training**
- Efficiently training models on a single GPU.
- Profiling tools and techniques to optimize performance.

[Read the Guide](../02_singlegpu_training/)

---

### 03. **Multi-GPU Training with Data Parallelism (DP)**
- Scaling models across multiple GPUs using `torch.nn.DataParallel`.
- Profiling and optimizing data parallel workloads.

[Read the Guide](../03_multigpu_dp_training/)

---

### 04. **Distributed Data Parallel (DDP) Training**
- Leveraging `torch.nn.parallel.DistributedDataParallel` for efficient multi-GPU training.
- Setting up process groups, distributed samplers, and profiling DDP workloads.

[Read the Guide](../04_multigpu_ddp_training/)

---

### 05. **Fully Sharded Data Parallel (FSDP) Training**
- Training large models with memory efficiency using Fully Sharded Data Parallel (FSDP).
- Fine-tuning large-scale models like CodeLlama with gradient checkpointing and parameter sharding.

[Read the Guide](./05_multigpu_fsdp_training/)

---

## **Highlights**

### Single-GPU Training
- Profiling tools like `line_profiler`.
- Optimizing data loaders and batch sizes.

### Multi-GPU Training
- Data Parallel (DP): Easy to implement but less efficient.
- Distributed Data Parallel (DDP): Faster, scales better, and reduces communication overhead.
- Fully Sharded Data Parallel (FSDP): Enables training of extremely large models by sharding parameters and gradients across GPUs.

### Advanced Features
- Gradient checkpointing for memory savings.
- Batch size tuning for performance optimization.
- Fine-tuning large-scale models like CodeLlama.

---

## **Getting Started**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-name/deep-learning-training-examples.git
   cd deep-learning-training-examples
   ```

2. **Set Up Conda Environment**:
   ```bash
   module load miniconda
   conda create -n deep_learning python=3.9 --yes
   conda activate deep_learning
   pip install torch torchvision transformers
   ```

3. **Navigate to a Training Guide**:
   ```bash
   cd 02_singlegpu_training
   python mnist_model.py
   ```

---

## **Resources**

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---

## **Contribute**

Contributions are welcome! Please submit a pull request or open an issue to suggest improvements or report bugs.

---

## **License**

This repository is licensed under the MIT License. See the `LICENSE` file for more details.

