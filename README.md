# Federated Optimization

This project is a course assignment for Convex Optimization, implementing the federated learning algorithm (FedAvg) as described in the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). The implementation focuses on the optimization aspects of distributed machine learning.

## Project Overview

This project demonstrates the application of optimization in distributed machine learning through federated learning. It implements:

1. **Centralized Training**: Standard neural network training with loss functions
2. **Federated Learning**: Distributed training with model aggregation
3. **Optimization Components**:
   - Gradient descent optimization
   - Model parameter averaging
   - Loss function minimization
   - Distributed optimization

## Features

- Support for both IID and non-IID data distributions
- Implementation of MLP and CNN models
- Experiments on MNIST and CIFAR10 datasets
- Visualization of training loss curves
- Performance comparison between centralized and federated learning

## Requirements
- Python >= 3.6
- PyTorch >= 0.4
- torchvision
- matplotlib
- numpy

## Implementation Details

### Centralized Training
Run the standard neural network training:
```bash
python main_nn.py --dataset mnist --model cnn --epochs 50 --gpu 0
```

### Federated Learning
Run the federated learning implementation:
```bash
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0
```

Key parameters:
- `--dataset`: Choose between 'mnist' or 'cifar'
- `--iid`: Use IID data distribution (omit for non-IID)
- `--model`: Choose between 'mlp' or 'cnn'
- `--epochs`: Number of training rounds
- `--gpu`: GPU device number
- `--all_clients`: Average over all client models

Note: For CIFAR-10, `num_channels` must be set to 3.

## Optimization Concepts

This implementation demonstrates several key concepts in optimization:
1. Distributed optimization
2. Gradient-based methods
3. Parameter averaging
4. Convergence analysis
5. Loss function properties


