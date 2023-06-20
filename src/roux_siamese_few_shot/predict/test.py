import numpy as np
from torch.autograd import Variable


def test(model, loader, args):
    model.eval()
    _all = []
    all_labels = []

    for batch_idx, (x, labels) in enumerate(loader):
        if args.cuda:
            x, labels = x.cuda(), labels.cuda()
        x, labels = Variable(x, volatile=True), Variable(labels)
        output = model.forward_once(x)
        _all.extend(output.data.cpu().numpy().tolist())
        all_labels.extend(labels.data.cpu().numpy().tolist())

    numpy_all = np.array(_all)
    numpy_labels = np.array(all_labels)
    return numpy_all, numpy_labels
