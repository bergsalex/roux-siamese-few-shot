import random
import itertools
import numpy as np

from roux_siamese_few_shot.datasets.utils import get_label_counts, label_start_index


def get_combinations_tuple(n, k):
    """Return a tuple of all combinations of k items from n items."""
    return tuple(itertools.combinations(range(n), k))


def create_omniglot_pairs(dataset, full=True):
    x0_data = []
    x1_data = []
    label = []

    # Each character has 20 examples
    n_examples_per_label = 20
    # The size of our dataset divided by the number of examples per label
    # gives us the
    n_labels = int(len(dataset) / n_examples_per_label)

    # We can create pairs of every combination of similar characters
    label_combs = get_combinations_tuple(n_examples_per_label, 2)

    for label_inc in range(0, n_labels, 20):
        # First we'll do pairs of the same class
        for comb in label_combs:
            idx_0 = label_inc + comb[0]
            idx_1 = label_inc + comb[1]
            data_0, label_0 = dataset[idx_0]
            data_1, label_1 = dataset[idx_1]

            assert label_0 == label_1

            x0_data.append(data_0)
            x1_data.append(data_1)
            label.append(1)

        # Now we'll do pairs of different classes
        # For each item in the class
        for idx in range(0, 20):
            count = 0
            # Let's pair it with _every_ other item in a different class
            # ... why not, right? ok... because it takes a really long time
            idx_0 = label_inc + idx

            for other_label_inc in range(0, n_labels, 20):
                if label_inc != other_label_inc:
                    for other_idx in range(0, 20):
                        idx_1 = other_label_inc + other_idx
                        data_0, label_0 = dataset[idx_0]
                        data_1, label_1 = dataset[idx_1]

                        assert label_0 != label_1

                        x0_data.append(data_0)
                        x1_data.append(data_1)
                        label.append(0)

                        if not full:
                            break

    label = np.array(label, dtype=np.int32)
    return x0_data, x1_data, label


def create_omniglot_pairs_alphabet(dataset, n_combs_per_label=None):
    x0_data = []
    x1_data = []
    label = []
    cursor = 0

    counts = get_label_counts(dataset)

    for label_inc in counts.keys():
        if label_inc > 0:
            cursor = cursor + counts[label_inc - 1]

        label_combs = list(
            itertools.combinations(range(cursor, cursor + counts[label_inc]), 2))
        if n_combs_per_label:
            random_selections = random.sample(label_combs, n_combs_per_label)
        else:
            random_selections = label_combs

        # First we'll do pairs of the same class
        for comb in random_selections:
            idx_0 = comb[0]
            idx_1 = comb[1]
            data_0, label_0 = dataset[idx_0]
            data_1, label_1 = dataset[idx_1]

            assert label_0 == label_1

            x0_data.append(data_0)
            x1_data.append(data_1)
            label.append(1)

        # Now we'll do pairs of different classes
        # For each item in the class
        other_labels = list(counts.keys())
        other_labels.remove(label_inc)

        label_and_index = [(label, label_start_index(counts, label)) for label in
                           other_labels]
        for idx_0 in range(cursor, cursor + counts[label_inc]):
            data_0, label_0 = dataset[idx_0]
            for o_label_inc, start_index in label_and_index:
                random_pairs = random.sample(
                    range(start_index, start_index + counts[o_label_inc]), 20)
                for pair in random_pairs:
                    data_1, label_1 = dataset[pair]

                    assert label_0 != label_1

                    x0_data.append(data_0)
                    x1_data.append(data_1)
                    label.append(0)

    label = np.array(label, dtype=np.int32)
    return x0_data, x1_data, label