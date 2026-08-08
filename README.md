# Brain Tumor Detection Using CNNs

A PyTorch-based convolutional neural network for classifying brain MRI images into two categories: brain tumor and no brain tumor.

## Overview

This project implements an image classification pipeline using a Convolutional Neural Network (CNN). The pipeline covers dataset preparation, image preprocessing, model training, model evaluation, and saving the trained model.

The input images are resized to 128 × 128 pixels and converted to tensors before being passed to the network.

## Model

The CNN consists of:

- Two convolutional layers
- ReLU activation functions
- Two max-pooling layers
- A flattening layer
- Two fully connected layers

The final layer produces two outputs corresponding to the two classes.

### Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Input size    | 128 × 128        |
| Batch size    | 16               |
| Epochs        | 20               |
| Optimizer     | Adam             |
| Learning rate | 0.001            |
| Loss function | CrossEntropyLoss |

## Dataset

The dataset contains 253 MRI images:

- 155 images with tumors
- 98 images without tumors

An 80:20 split was used:

- Training: 202 images
- Testing: 51 images

The dataset itself is excluded from this repository.

## Results

The model achieved a test accuracy of **88.24%** on the held-out test set.

## Project Structure

```text
tumor-detect-cnn/
├── src/
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── load_dataset.py
│   ├── split_dataset.py
│   └── test_model.py
├── models/
├── .gitignore
└── README.md
```

## Usage

### 1. Install dependencies

Create and activate a virtual environment, then install the required packages:

```bash
pip install torch torchvision
```

### 2. Prepare the dataset

Place the source dataset in the location expected by split_dataset.py, then run:

```bash
python src/split_dataset.py
```

### 3. Train the model

```bash
python src/train.py
```

The trained model is saved in the models/ directory.

### 4. Evaluate the model

```bash
python src/test.py
```

### Technologies

- Python
- PyTorch
- TorchVision
- Git

### Limitations

The model is trained on a relatively small dataset and performs binary image classification. Its predictions should not be interpreted as medical diagnoses.

### References

- [PyTorch](https://pytorch.org/)
- [TorchVision](https://docs.pytorch.org/vision/stable/)
- [Python](https://www.python.org/)
- [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/arwabasal/brain-tumor-mri-detection?utm_source=chatgpt.com)
