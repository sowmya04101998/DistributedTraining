### Assignment: Training a CNN on the CIFAR-10 Dataset Using DataParallelism

#### **Objective:**
Train a Convolutional Neural Network (CNN) on the **CIFAR-10 dataset** using PyTorch, and implement **data parallelism** to utilize multiple GPUs for faster training.

---

#### **Assignment Description:**
The goal is to build and train a CNN on the CIFAR-10 dataset using **PyTorch's DataParallel** module for multi-GPU support. The dataset consists of 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images.

---

### **Steps to Complete:**

1. **Load and Preprocess the CIFAR-10 Dataset**:
   - Use the `torchvision.datasets` library to load the CIFAR-10 dataset.
   - Normalize the dataset using the mean and standard deviation of the CIFAR-10 dataset.

2. **Define the CNN Model**:
   - Create a CNN with at least three convolutional layers and two fully connected layers.
   - Use activation functions, dropout, and max pooling.

3. **Implement DataParallelism**:
   - Check if multiple GPUs are available.
   - Wrap the model using `torch.nn.DataParallel` to utilize all available GPUs.

4. **Train the Model**:
   - Train the CNN on the CIFAR-10 dataset for a specified number of epochs.
   - Use a suitable optimizer (e.g., Adam) and learning rate scheduler.

5. **Evaluate the Model**:
   - Evaluate the model's performance on the test dataset.
   - Calculate and display the test loss and accuracy.

6. **Submit Results**:
   - Provide the training and testing accuracy for the model.
   - Save the trained model and submit the code with comments.
