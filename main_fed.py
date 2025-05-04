#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from typing import Dict, List, Tuple, Optional

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


class FederatedLearning:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.dataset_train = None
        self.dataset_test = None
        self.dict_users = None
        self.net_glob = None
        self.w_glob = None

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
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset_train = datasets.MNIST(
            '/Users/bob/Downloads/federated-learning-master/data/mnist',
            train=True,
            download=True,
            transform=trans_mnist
        )
        self.dataset_test = datasets.MNIST(
            '/Users/bob/Downloads/federated-learning-master/data/mnist',
            train=False,
            download=True,
            transform=trans_mnist
        )
        self.dict_users = mnist_iid(self.dataset_train, self.args.num_users) if self.args.iid else \
                         mnist_noniid(self.dataset_train, self.args.num_users)

    def _load_cifar(self) -> None:
        """Load CIFAR10 dataset"""
        trans_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset_train = datasets.CIFAR10(
            '/Users/bob/Downloads/federated-learning-master/data/cifar',
            train=True,
            download=True,
            transform=trans_cifar
        )
        self.dataset_test = datasets.CIFAR10(
            '/Users/bob/Downloads/federated-learning-master/data/cifar',
            train=False,
            download=True,
            transform=trans_cifar
        )
        if self.args.iid:
            self.dict_users = cifar_iid(self.dataset_train, self.args.num_users)
        else:
            raise ValueError('Only IID setting is supported for CIFAR10')

    def build_model(self) -> None:
        """Build global model"""
        img_size = self.dataset_train[0][0].shape
        
        if self.args.model == 'cnn' and self.args.dataset == 'cifar':
            self.net_glob = CNNCifar(args=self.args).to(self.device)
        elif self.args.model == 'cnn' and self.args.dataset == 'mnist':
            self.net_glob = CNNMnist(args=self.args).to(self.device)
        elif self.args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            self.net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=self.args.num_classes).to(self.device)
        else:
            raise ValueError('Unrecognized model')
        
        print(self.net_glob)
        self.net_glob.train()
        self.w_glob = self.net_glob.state_dict()

    def train(self) -> List[float]:
        """Execute federated learning training process"""
        loss_train = []
        
        if self.args.all_clients:
            print("Aggregation over all clients")
            w_locals = [self.w_glob for _ in range(self.args.num_users)]
        
        for iter in range(self.args.epochs):
            loss_locals = []
            if not self.args.all_clients:
                w_locals = []
            
            # Select clients for training
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
            
            # Local training
            for idx in idxs_users:
                local = LocalUpdate(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(self.net_glob).to(self.device))
                
                if self.args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            
            # Update global weights
            self.w_glob = FedAvg(w_locals)
            self.net_glob.load_state_dict(self.w_glob)
            
            # Print training information
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
        
        return loss_train

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model performance"""
        self.net_glob.eval()
        acc_train, loss_train = test_img(self.net_glob, self.dataset_train, self.args)
        acc_test, loss_test = test_img(self.net_glob, self.dataset_test, self.args)
        return acc_train, acc_test

    def plot_loss(self, loss_train: List[float]) -> None:
        """Plot training loss curve"""
        plt.figure()
        plt.plot(range(len(loss_train)), loss_train)
        plt.ylabel('train_loss')
        plt.savefig('/Users/bob/Downloads/federated-learning-master/save/fed_{}_{}_{}_C{}_iid{}.png'.format(
            self.args.dataset, self.args.model, self.args.epochs, self.args.frac, self.args.iid
        ))


def main():
    # Parse arguments
    args = args_parser()
    
    # Initialize federated learning
    fl = FederatedLearning(args)
    
    # Load dataset
    fl.load_dataset()
    
    # Build model
    fl.build_model()
    
    # Train model
    loss_train = fl.train()
    
    # Plot loss curve
    fl.plot_loss(loss_train)
    
    # Evaluate model
    acc_train, acc_test = fl.evaluate()
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))


if __name__ == '__main__':
    main()

