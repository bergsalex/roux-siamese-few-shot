import torch
import numpy as np
import torchvision.datasets as dsets

from net import SiameseNetwork
from torchvision import transforms
from torch.utils.data import DataLoader

from roux_siamese_few_shot.plot import plot_mnist
from roux_siamese_few_shot.args import parse_args
from roux_siamese_few_shot.train import training_run
from roux_siamese_few_shot.predict.test import test
from roux_siamese_few_shot.datasets.pairs.mnist import create_pairs_rand
from roux_siamese_few_shot.datasets.pairs.dataset import Dataset


def create_iterator(data, label, batchsize, shuffle=False):
    digit_indices = [np.where(label == i)[0] for i in range(10)]
    x0, x1, label = create_pairs_rand(data, digit_indices)
    ret = Dataset(x0, x1, label)
    return ret


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("Args: %s" % args)

    # create pair dataset iterator
    train_dataset = dsets.MNIST(
        root='../data/',
        train=True,
        # transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ]),
        download=True
    )
    test_dataset = dsets.MNIST(
        root='../data/',
        train=False,

        # XXX ToTensor scale to 0-1
        transform=transforms.Compose([
            transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    train_iter = create_iterator(
        train_dataset.train_data.numpy(),
        train_dataset.train_labels.numpy(),
        args.batchsize)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        train_iter,
        batch_size=args.batchsize, shuffle=True, **kwargs)

    if len(args.model) == 0:
        model = training_run(SiameseNetwork(), train_loader, args)

    else:
        saved_model = torch.load(args.model)
        model = SiameseNetwork()
        model.load_state_dict(saved_model)
        if args.cuda:
            model.cuda()

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batchsize,
                                              shuffle=True,
                                              **kwargs)

    numpy_all, numpy_labels = test(model, test_loader, args)
    plot_mnist(numpy_all, numpy_labels)


if __name__ == '__main__':
    main()
