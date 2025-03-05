# High-Performance Deep Learning with SLURM

## Overview
This repository provides a comprehensive guide to deep learning with **PyTorch**, along with best practices for running workloads on an **HPC cluster using SLURM**. It includes:
- **Deep Learning Basics**: Jupyter notebooks covering foundational concepts.
- **SLURM Job Scheduling**: Guides and scripts for distributed training.
- **Module Management**: Best practices for handling dependencies on HPC clusters.

---

## Repository Structure
```
/01_introduction/
 ├── Modules.md                  # Guide on managing modules
 ├── README.md                   # Project documentation
 ├── SLURM.md                    # SLURM job scheduling guide
 ├── introduction_to_DeepLearning.ipynb  # Jupyter Notebook on DL basics
 ├── slurm_cheatbook.pdf         # SLURM command reference
```

---

## Contents
### 🔹 Deep Learning Topics Covered
  - **Understanding Tensors** in PyTorch
  - **Forward & Backward Propagation**
  - **Loss Functions & Optimization**
  - **Leveraging PyTorch Tensor Cores**
  - **Building a Simple Neural Network**

### 🔹 SLURM & HPC Topics Covered
  - **Managing Job Queues & Partitions**
  - **Writing & Submitting SLURM Jobs**
  - **Monitoring & Debugging Jobs**
  - **Using SLURM for Distributed Training**
  - **Managing Dependencies with Modules**

### Prerequisites
To effectively use this repository, ensure you have:
- **Python basics**
- **Familiarity with NumPy & PyTorch**
- **Access to an HPC cluster** (if using SLURM)



## Running Deep Learning Jobs on SLURM
This repository includes guides on **efficiently executing deep learning jobs on an HPC cluster using SLURM**.

📌 **SLURM.md** – Full SLURM job scheduling guide.
📌 **SLURM Cheatbook** – Quick SLURM reference.
📌 **Modules.md** – Managing dependencies on an HPC cluster.

### 🔹 Submitting a SLURM Job
Submit a job using:
```bash
sbatch my_slurm_script.sh
```

Monitor job status:
```bash
squeue -u <your-username>
```

Cancel a job if needed:
```bash
scancel <job_id>
```

---

## Additional Resources
📚 [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
📚 [SLURM Official Guide](https://slurm.schedmd.com/documentation.html)
📚 [Deep Learning Book by Ian Goodfellow](https://www.deeplearningbook.org/)

For questions or contributions, feel free to **open an issue** or **submit a pull request**.

