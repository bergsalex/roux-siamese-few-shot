import random
import numpy as np


def create_pairs(data, digit_indices):
    """Creates a contrasting example of each class."""
    x0_data = []
    x1_data = []
    label = []

    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        # make n pairs with each number
        for i in range(n):
            # make pairs of the same class
            # label is 1
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            # scale data to 0-1
            # XXX this does ToTensor also
            x0_data.append(data[z1] / 255.0)
            x1_data.append(data[z2] / 255.0)
            label.append(1)

            # make pairs of different classes
            # since the minimum value is 1, it is not the same class
            # label is 0
            for inc in range(1, 10):
                dn = (d + inc) % 10
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                # scale data to 0-1
                # XXX this does ToTensor also
                x0_data.append(data[z1] / 255.0)
                x1_data.append(data[z2] / 255.0)
                label.append(0)

    x0_data = np.array(x0_data, dtype=np.float32)
    x0_data = x0_data.reshape([-1, 1, 28, 28])
    x1_data = np.array(x1_data, dtype=np.float32)
    x1_data = x1_data.reshape([-1, 1, 28, 28])
    label = np.array(label, dtype=np.int32)
    return x0_data, x1_data, label


def create_pairs_rand(data, digit_indices):
    """This is the original create pairs function."""
    x0_data = []
    x1_data = []
    label = []

    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        # make n pairs with each number
        for i in range(n):
            # make pairs of the same class
            # label is 1
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            # scale data to 0-1
            # XXX this does ToTensor also
            x0_data.append(data[z1] / 255.0)
            x1_data.append(data[z2] / 255.0)
            label.append(1)

            # make pairs of different classes
            # since the minimum value is 1, it is not the same class
            # label is 0
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # scale data to 0-1
            # XXX this does ToTensor also
            x0_data.append(data[z1] / 255.0)
            x1_data.append(data[z2] / 255.0)
            label.append(0)

    x0_data = np.array(x0_data, dtype=np.float32)
    x0_data = x0_data.reshape([-1, 1, 28, 28])
    x1_data = np.array(x1_data, dtype=np.float32)
    x1_data = x1_data.reshape([-1, 1, 28, 28])
    label = np.array(label, dtype=np.int32)
    return x0_data, x1_data, label
