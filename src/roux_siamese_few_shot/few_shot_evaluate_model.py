import torch
from torch.utils.data import DataLoader
from roux_siamese_few_shot.net import SiameseNetwork
from roux_siamese_few_shot.few_shot.dataset import FewSh


def load_model_from_file(model_inst, path):
    model_inst.load_state_dict(torch.load(path))
    return model_inst


def evaluate(model_path, test_dataset,
             way=5, shot=5,
             support_set=None, query_set=None):

    model = load_model_from_file(SiameseNetwork(), model_path)

    fs_dataset = FewSh(test_dataset, way, shot, support_set, query_set)

    query_loader = DataLoader(fs_dataset.query,
                              batch_size=128,
                              shuffle=True)