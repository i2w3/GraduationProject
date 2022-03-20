import os
import time
import torch
import numpy as np
import torch.nn as nn
from d2l import torch as d2l
from datetime import datetime
import matplotlib.pyplot as plt


def check_Device():
    # 检查cuda加速
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def seed_Setting(SEED):
    # 根据SEED设置随机种子，增加PyTorch中模型的可复现性
    if SEED:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # 不需要复现则启用cudnn的benchmark提高性能
    else:
        torch.backends.cudnn.benchmark = True


def get_accuracy(net, data_iter, device=None):
    # 计算模型在数据集上的精度
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    # 训练损失总和，词元数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(train_loader, model, criterion, optimizer, device):
    # 模型训练
    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    # 模型验证

    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    # 模型的训练循环

    print(f'{datetime.now().time().replace(microsecond=0)} --- '
          f'Start training loop\t'
          f'training on: {device}')  # 打印训练设备

    # 保存数据
    train_acces = []
    valid_acces = []
    train_losses = []
    valid_losses = []

    # 开始循环训练模型
    for epoch in range(0, epochs):

        # 训练一次
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # 验证一次
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            # 输出本次训练的数据
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch + 1}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
            train_acces.append(train_acc)
            valid_acces.append(valid_acc)

    # 循环训练结束，计算top1err和top5err，根据时间戳保存数据并绘图
    unix_timestamp = str(int(time.time()))
    top1err = evaluteTop1(model, valid_loader)
    top5err = evaluteTop5(model, valid_loader)
    print(f'{datetime.now().time().replace(microsecond=0)} --- '
          f'当前时间戳为: {unix_timestamp}\t'
          f'top1 err.: {top1err}\t'
          f'top5 err.: {top5err}')
    full_plot(epochs, train_losses, valid_losses, train_acces, valid_acces, unix_timestamp)
    saveNpy(unix_timestamp, train_losses, valid_losses, train_acces, valid_acces)
    return model, optimizer, (train_losses, valid_losses), (train_acces, valid_acces)


def evaluteTop1(model, loader):
    # 计算top1 error
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(check_Device()), y.to(check_Device())
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def evaluteTop5(model, loader):
    # 计算top5 error
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(check_Device()), y.to(check_Device())
        with torch.no_grad():
            logits = model(x)
            maxk = max((1, 5))

            y_resize = y.view(-1, 1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().float().item()
    return correct / total


def full_plot(N_EPOCHS, train_loss, valid_loss, train_acc, valid_acc, unix_timestamp, savePath='./png/'):
    # 绘图
    train_loss = np.array(train_loss)
    if valid_loss:
        # 不一定绘制valid_loss
        valid_loss = np.array(valid_loss)
    train_acc = np.array(train_acc)
    valid_acc = np.array(valid_acc)

    fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)  # 创建一个包含一个axes的figure

    l1 = ax.plot(train_loss, '--', color='blue', label="Train loss")
    if valid_loss:
        l2 = ax.plot(valid_loss, '--', color='red', label="Valid loss")

    ax2 = ax.twinx()

    l3 = ax2.plot(train_acc, color='blue', label="Train acc")
    l4 = ax2.plot(valid_acc, color='red', label="Valid acc")

    if valid_loss:
        lns = l1 + l2 + l3 + l4
    else:
        lns = l1 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')

    if N_EPOCHS == 50:
        blink = 7
    else:
        blink = 2
    ticks = list(range(0, N_EPOCHS, blink))
    ticks.append(N_EPOCHS - 1)
    tickl = [i + 1 for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tickl, rotation=10)

    ax.set(title="Acc & Loss over Epochs", xlabel='Epoch', ylabel="Loss")

    ax2.set(ylabel="Acc")

    ax.grid(b=False, axis="y")
    ax2.grid(b=False, axis="y")
    ax.grid(b=True, axis="x")

    Path = savePath + str(unix_timestamp)
    if not os.path.exists(Path):
        os.makedirs(Path)
    fig.savefig(Path + "/" + "fig.png")
    fig.show()
    plt.style.use('default')


def loadNpy(unix_timestamp, Path="./png/"):
    # 读取训练数据
    npy_path = Path + str(unix_timestamp) + "/"

    train_acc = np.load(npy_path + "train_acc.npy", encoding="latin1")
    valid_acc = np.load(npy_path + "valid_acc.npy", encoding="latin1")

    train_loss = np.load(npy_path + "train_loss.npy", encoding="latin1")
    valid_loss = np.load(npy_path + "valid_loss.npy", encoding="latin1")

    return (train_acc, valid_acc), (train_loss, valid_loss)


def saveNpy(unix_timestamp, train_loss, valid_loss, train_acc, valid_acc, Path="./png/"):
    # 保存训练数据
    npy_path = Path + str(unix_timestamp) + "/"

    np.save(npy_path + "train_acc" + ".npy", train_acc)
    np.save(npy_path + "valid_acc" + ".npy", valid_acc)
    np.save(npy_path + "train_loss" + ".npy", train_loss)
    np.save(npy_path + "valid_loss" + ".npy", valid_loss)
