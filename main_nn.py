#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from typing import Tuple, List

from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar


class NeuralNetwork:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.dataset_train = None
        self.dataset_test = None
        self.net_glob = None
        torch.manual_seed(args.seed)

    def load_dataset(self) -> None:
        """Load and preprocess dataset"""
        if self.args.dataset == 'mnist':
            self._load_mnist()
        elif self.args.dataset == 'cifar':
            self._load_cifar()
        else:
            raise ValueError('Unrecognized dataset')

    def _load_mnist(self) -> None:
        """Load MNIST dataset"""
        self.dataset_train = datasets.MNIST(
            '/Users/bob/Downloads/federated-learning-master/data/mnist/',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        img_size = self.dataset_train[0][0].shape
        return img_size

    def _load_cifar(self) -> None:
        """Load CIFAR10 dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset_train = datasets.CIFAR10(
            '/Users/bob/Downloads/federated-learning-master/data/cifar',
            train=True,
            transform=transform,
            target_transform=None,
            download=True
        )
        img_size = self.dataset_train[0][0].shape
        return img_size

    def build_model(self, img_size) -> None:
        """Build neural network model"""
        if self.args.model == 'cnn' and self.args.dataset == 'cifar':
            self.net_glob = CNNCifar(args=self.args).to(self.device)
        elif self.args.model == 'cnn' and self.args.dataset == 'mnist':
            self.net_glob = CNNMnist(args=self.args).to(self.device)
        elif self.args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            self.net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=self.args.num_classes).to(self.device)
        else:
            raise ValueError('Unrecognized model')
        print(self.net_glob)

    def train(self) -> List[float]:
        """Execute training process"""
        optimizer = optim.Adam(self.net_glob.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        train_loader = DataLoader(self.dataset_train, batch_size=64, shuffle=True)

        list_loss = []
        self.net_glob.train()
        
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.net_glob(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                batch_loss.append(loss.item())
            
            loss_avg = sum(batch_loss) / len(batch_loss)
            print('\nTrain loss:', loss_avg)
            list_loss.append(loss_avg)
        
        return list_loss

    def test(self) -> Tuple[int, float]:
        """Evaluate model performance"""
        self.net_glob.eval()
        test_loss = 0
        correct = 0
        test_loader = self._get_test_loader()
        
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            log_probs = self.net_glob(data)
            test_loss += F.cross_entropy(log_probs, target).item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        return correct, test_loss

    def _get_test_loader(self) -> DataLoader:
        """Get test data loader"""
        if self.args.dataset == 'mnist':
            dataset_test = datasets.MNIST(
                './data/mnist/',
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )
        elif self.args.dataset == 'cifar':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset_test = datasets.CIFAR10(
                './data/cifar',
                train=False,
                transform=transform,
                target_transform=None,
                download=True
            )
        else:
            raise ValueError('Unrecognized dataset')
        
        return DataLoader(dataset_test, batch_size=1000, shuffle=False)

    def plot_loss(self, list_loss: List[float]) -> None:
        """Plot training loss curve"""
        plt.figure()
        plt.plot(range(len(list_loss)), list_loss)
        plt.xlabel('epochs')
        plt.ylabel('train loss')
        plt.savefig('./log/nn_{}_{}_{}.png'.format(
            self.args.dataset, self.args.model, self.args.epochs
        ))


def main():
    # Parse arguments
    args = args_parser()
    
    # Initialize neural network
    nn = NeuralNetwork(args)
    
    # Load dataset
    nn.load_dataset()
    
    # Build model
    img_size = nn.dataset_train[0][0].shape
    nn.build_model(img_size)
    
    # Train model
    list_loss = nn.train()
    
    # Plot loss curve
    nn.plot_loss(list_loss)
    
    # Test model
    print('Test on', len(nn.dataset_test), 'samples')
    nn.test()


if __name__ == '__main__':
    main()
