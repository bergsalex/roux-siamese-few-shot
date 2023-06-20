import torch
import torchvision
from torch.utils.data import DataLoader

from roux_siamese_few_shot.net import SiameseNetwork
from roux_siamese_few_shot.args import parse_args
from roux_siamese_few_shot.train import training_run

from roux_siamese_few_shot.datasets.pairs.dataset import Dataset
from roux_siamese_few_shot.datasets.pairs.omniglot import create_omniglot_pairs

from roux_siamese_few_shot.few_shot.dataset import FewShotDataset
from roux_siamese_few_shot.predict import closest_support, closest_prototype


def main():
    args = parse_args()

    omniglot_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize([28, 28], antialias=True)
    ])

    omniglot_dataset = torchvision.datasets.Omniglot(root='data',
                                                     background=True,
                                                     download=True,
                                                     transform=omniglot_transform)

    pairs = create_omniglot_pairs(omniglot_dataset)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(Dataset(*pairs),
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               **kwargs)

    model = training_run(SiameseNetwork(), train_loader, args)

    omniglot_test_dataset = torchvision.datasets.Omniglot(root='data', background=False,
                                                          download=True,
                                                          transform=omniglot_transform)

    few_shot_dataset = FewShotDataset(omniglot_test_dataset, way=5, shot=5)

    closest_support_accuracy = closest_support.calculate_accuracy(model,
                                                                  few_shot_dataset)

    print(f"Closest support accuracy: {closest_support_accuracy}")

    closest_prototype_accuracy = closest_prototype.calculate_accuracy(model,
                                                                      few_shot_dataset)

    print(f"Closest prototype accuracy: {closest_prototype_accuracy}")
