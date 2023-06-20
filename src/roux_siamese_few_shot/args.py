import torch
import argparse


class MainArgs:
    """The main arguments schema for model training."""

    def __init__(self, e=5, b=128, no_cuda=False, m='', train_plot=False,
                 learning_rate=0.01, momentum=0.5):
        self.epoch = e
        self.batchsize = b
        self.no_cuda = no_cuda
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        self.model = m
        self.train_plot = train_plot
        self.learning_rate = learning_rate
        self.momentum = momentum

    def __str__(self):
        return ' '.join([f"{k}={v}" for k, v in self.__dict__.items()])


def parse_args():
    """Parse the arguments for model training from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', '-m', default='',
                        help='Give a model to test')
    parser.add_argument('--train-plot', action='store_true', default=False,
                        help='Plot train loss')
    args = parser.parse_args([f'-e={e}', f'-b={b}'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
