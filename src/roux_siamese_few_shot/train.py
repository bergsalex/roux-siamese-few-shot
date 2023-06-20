import uuid
import time
import torch
import pathlib
from torch.autograd import Variable
import matplotlib.pyplot as plt

from roux_siamese_few_shot.contrastive import ContrastiveLoss


def train(model, loader, criterion, optimizer, args, epoch, train_id):
    train_loss = []
    model.train()
    start = time.time()
    start_epoch = time.time()
    for batch_idx, (x0, x1, labels) in enumerate(loader):
        labels = labels.float()
        if args.cuda:
            x0, x1, labels = x0.cuda(), x1.cuda(), labels.cuda()
        x0, x1, labels = Variable(x0), Variable(x1), Variable(labels)
        output1, output2 = model(x0, x1)
        loss = criterion(output1, output2, labels)
        train_loss.append(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = []

        for idx, logit in enumerate([output1, output2]):
            corrects = (torch.max(logit, 1)[1].data == labels.long().data).sum()
            accu = float(corrects) / float(labels.size()[0])
            accuracy.append(accu)

        if batch_idx % args.batchsize == 0:
            end = time.time()
            took = end - start
            for idx, accu in enumerate(accuracy):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}\tTook: {:.2f}\tOut: {}\tAccu: {:.2f}'.format(
                    epoch, batch_idx * len(labels), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.data.item(),
                    took, idx, accu * 100.))
            start = time.time()
    torch.save(model.state_dict(), f'model_training/{train_id}/model-epoch-{epoch}.pth')
    end = time.time()
    took = end - start_epoch
    print('Train epoch: {} \tTook:{:.2f}'.format(epoch, took))
    return train_loss


def training_run(model, train_loader, args, learning_rate=0.01, momentum=0.9):
    if args.cuda:
        model.cuda()

    criterion = ContrastiveLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum)

    train_loss = []
    train_id = uuid.uuid4()
    pathlib.Path(f'model_training/{train_id}/').mkdir(parents=True, exist_ok=True)
    print(f'Started training for train run with id: {train_id}')
    for epoch in range(1, args.epoch + 1):
        train_loss.extend(train(model,
                                train_loader,
                                criterion,
                                optimizer,
                                args,
                                epoch,
                                train_id))
    print(f'Completed training for train run with id: {train_id}')

    if args.train_plot:
        plt.gca().cla()
        plt.plot(train_loss, label="train loss")
        plt.legend()
        plt.draw()
        plt.savefig('train_loss.png')
        plt.gca().clear()

    return model
