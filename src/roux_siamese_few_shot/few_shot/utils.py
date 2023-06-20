import random
from typing import List, Tuple

LIST_OF_TUPLES = List[Tuple[int, int]]


def split_query_and_support(data_loader,
                            selected_classes: List[int],
                            shot: int,
                            counts: dict) -> Tuple[LIST_OF_TUPLES, LIST_OF_TUPLES]:
    labels = {k: {'support': [], 'query': []} for k in counts.keys() if
              k in selected_classes}
    for idx in range(len(data_loader)):
        _, _label = data_loader[idx]
        if _label in selected_classes:
            labels[_label]['query'].append(idx)

    for label in labels:
        labels[label]['support'] = list(random.sample(labels[label]['query'], shot))
        for s in labels[label]['support']:
            labels[label]['query'].remove(s)

    query = [(idx, label)
             for label, values in labels.items()
             for idx in values['query']]
    support = [(idx, label)
               for label, values in labels.items()
               for idx in values['support']]

    return query, support
