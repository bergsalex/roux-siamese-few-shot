import torch
from roux_siamese_few_shot.datasets.utils import get_label_counts
from roux_siamese_few_shot.few_shot.utils import split_query_and_support


class FewShotDataloader:

    def __init__(self, data_loader, set_idxs):
        self.size = len(set_idxs)
        self._counts = get_label_counts(data_loader)
        self._data_loader = data_loader
        self._idx_set = set_idxs

    def __getitem__(self, index):
        return self._data_loader[self._idx_set[index][0]]

    def __len__(self):
        return self.size


class FewShotDataset:

    def __init__(self, dataset, way=5, shot=5, support=None, query=None):
        self.dataset = dataset
        self.way = way
        self.shot = shot

        self.counts = get_label_counts(dataset)

        self._unique_labels = torch.tensor(list(self.counts.keys()))
        self.selected_classes = self._unique_labels[
            torch.randperm(len(self._unique_labels))[:self.way]]

        if support is not None and query is not None:
            self.support = FewShotDataloader(self.dataset, support)
            self.query = FewShotDataloader(self.dataset, query)
        else:
            q, s = split_query_and_support(self.dataset, self.selected_classes,
                                           self.shot, self.counts)

            self.query = FewShotDataloader(self.dataset, q)
            self.support = FewShotDataloader(self.dataset, s)
