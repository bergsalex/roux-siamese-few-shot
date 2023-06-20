import torch


def get_support_embeddings(model, support_loader, device='cuda'):
    with torch.no_grad():
        for images, labels in support_loader:
            images = images.to(device)
            embeddings = model.forward_once(images)
    return embeddings, labels


def classify_qvs(model, s_embeddings, qimage, device="cuda"):
    def euclidean_dist(x, y):
        return ((x - y) ** 2).sum(dim=1)

    s_embeddings, s_labels = s_embeddings

    with torch.no_grad():
        model.to(device)
        qimage.to(device)
        embedding = model.forward_once(qimage.unsqueeze(0))

        distances = [euclidean_dist(embedding, s_embedding)
                     for s_embedding in s_embeddings]
        distances = torch.tensor(distances)
        minidx = torch.argmin(distances)
    return s_labels[minidx]


def calculate_accuracy(model, fs_dataset, device='cuda'):
    correct = 0
    total = 0

    with torch.no_grad():
        support_embeddings = get_support_embeddings(model, fs_dataset.support)

        for images, labels in fs_dataset.query:
            images = images.to(device)
            labels = labels.to(device)
            for i in range(len(labels)):
                image = images[i]
                label = labels[i]
                prediction = classify_qvs(model, support_embeddings, image)
                if prediction == label.item():
                    correct += 1
                total += 1

    accuracy = correct / total

    return accuracy
