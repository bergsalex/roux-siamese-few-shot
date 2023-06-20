import torch


def create_prototype_vector(model, loader, way, device, selected_classes):
    prototype_vectors = torch.zeros(way, 2, device=device)
    counts = torch.zeros(way, device=device)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            embeddings = model.forward_once(images)
            for i, _class in enumerate(selected_classes):
                mask = (labels == _class)
                if mask.any():
                    prototype_vectors[i] += embeddings[mask].sum(dim=0)
                    counts[i] += mask.sum()

    prototype_vectors /= counts.unsqueeze(1)

    return prototype_vectors


def classify(model, image, prototype_vectors, device):
    with torch.no_grad():
        image = image.to(device)
        # Add batch dimension
        embedding = model.forward_once(image.unsqueeze(0))
        # Euclidean distance
        distances = ((prototype_vectors - embedding) ** 2).sum(dim=1)
        return distances.argmin().item()


def calculate_accuracy(model, fs_dataset, device='cuda'):
    correct = 0
    total = 0

    selected_classes = fs_dataset.selected_classes
    prototype_vectors = create_prototype_vector(model,
                                                fs_dataset.support,
                                                fs_dataset.way,
                                                device,
                                                selected_classes)

    with torch.no_grad():
        for images, labels in fs_dataset.query:
            images = images.to(device)
            labels = labels.to(device)
            for i in range(len(labels)):
                image = images[i]
                label = labels[i]
                prediction = selected_classes[classify(model,
                                                       image,
                                                       prototype_vectors,
                                                       device)]
                if prediction == label.item():
                    correct += 1
                total += 1

    accuracy = correct / total
    return accuracy
