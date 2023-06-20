
def get_label_counts(data_loader) -> dict:
    counts = {}
    for _, label in data_loader:
        try:
            counts[label] = counts[label] + 1
        except KeyError:
            counts[label] = 1
    return counts


def label_start_index(counts, label) -> int:
    index = 0

    if label != 0:
        for i in range(label):
            index = index + counts[i]

    return index
