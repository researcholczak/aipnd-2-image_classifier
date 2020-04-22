import myfunc as mf
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sb
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile, Image
from torchvision import datasets, transforms, models

def args_paser():
    paser = argparse.ArgumentParser(description='Neural network trainer')

    paser.add_argument('--data_dir', type=str, default=r'\flowers', help='data set directory')
    paser.add_argument('--cuda', type=bool, default='True', help='True: cuda, False: cpu')
    paser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    paser.add_argument('--n_epochs', type=int, default=2, help='number of epochs')
    paser.add_argument('--batch_size', type=int, default=20, help='dataloader batch size')
    paser.add_argument('--arch', type=str, default='vgg16', help='network architecture: vgg16 or alexnet')
    paser.add_argument('--n_hidden', type=str, default=512, help='network architecture: vgg16 or alexnet')
    paser.add_argument('--checkpoint_path', type=str, default='checkpoint_cmd.pth', help='save train model to file')
    args = paser.parse_args()
    return args

def main():

    args = args_paser()

    # Check if CUDA is available
    device = mf.set_device(use_gpu = args.cuda)

    # Data loading
    path_dir = r"..\data\AIP" + args.data_dir
    train_dir = path_dir + r'\train'
    valid_dir = path_dir + r'\valid'
    test_dir = path_dir + r'\test'

    # TODO: Build and train your network
    dataloaders, class_to_idx = mf.process_data(train_dir, valid_dir, test_dir, args.batch_size)
    model = mf.create_base_model(args.arch)
    model = mf.set_classifier(model, args.n_hidden)
    # if GPU is available, move the model to GPU
    model.to(device)
    # Loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()
    # Stochastic gradient descent optimizer
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.lr)
    # Train model
    model, optimizer = mf.train(model, optimizer, criterion, dataloaders['train'], dataloaders['valid'], len(class_to_idx), device, n_epochs = args.n_epochs)
    mf.test(model, criterion, dataloaders['test'], device, len(class_to_idx))
    print("YAY")
    #model.class_to_idx = class_to_idx
    # Save model to checkpoint
    mf.save_checkpoint(model, optimizer, class_to_idx, checkpoint = args.checkpoint_path)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
