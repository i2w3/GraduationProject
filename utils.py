import os
import sys
import time
import torch
import zipfile
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


def real_Time():
    # 获取当前时间(格式H:M:S)
    return datetime.now().time().replace(microsecond=0)


def get_Accuracy(net, data_iter, device=None):
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


def training_loop(model, criterion, optimizer, train_loader, valid_loader, device, epochs, print_every=1):
    # 模型的训练循环
    # milestones = [65, 110]
    # milestones = [80, 120]
    milestones = [50, 80]
    print(f'{real_Time()} --- '
          f'Start training loop\t'
          f'training on: {device}')  # 打印训练设备
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.05)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
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
            print(f'{real_Time()} --- '
                  f'Epoch: {epoch + 1}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
            train_acces.append(train_acc)
            valid_acces.append(valid_acc)

            old_lr = remove_extra_zero(get_LearningRate(optimizer))
            scheduler.step()
            new_lr = remove_extra_zero(get_LearningRate(optimizer))
            if old_lr != new_lr:
                print(f'\t\t\t\t\t\t\t\t'
                      f'Learning Rate from {old_lr:.3f} to {new_lr:.3f}')

    # 循环训练结束，计算top1acc和top5acc，根据时间戳保存数据并绘图
    unix_timestamp = str(int(time.time()))
    top1acc = evaluteTop1(model, valid_loader)
    top5acc = evaluteTop5(model, valid_loader)
    top1err, top5err = TopErr(model, valid_loader, device)
    print(f'{real_Time()} --- '
          f'Time Stamp: {unix_timestamp}\t\t'
          f'top1 acc: {top1acc}\t'
          f'top5 acc: {top5acc}')
    print(f'\t\t\t\t\t\t\t\t'
          f'top1 err: {top1err:.4f}\t'
          f'top5 err: {top5err:.4f}')
    full_Plot(train_losses, valid_losses, train_acces, valid_acces, unix_timestamp)
    save_Npy(unix_timestamp, train_losses, valid_losses, train_acces, valid_acces)
    save_Model(unix_timestamp, model)
    return model, optimizer


def evaluteTop1(model, loader):
    # 计算top1 acc
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
    # 计算top5 acc
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


def top_Err(model, loader, device):
    model.eval()

    # top1 acc 和 top5 acc
    correct_1 = 0.0
    correct_5 = 0.0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(loader):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top1
            correct_1 += correct[:, :1].sum()

    return 1 - correct_1 / len(loader.dataset), 1 - correct_5 / len(loader.dataset)


def full_Plot(train_loss, valid_loss, train_acc, valid_acc, unix_timestamp, savePath='./png/'):
    # 绘图，将train_loss, valid_loss, train_acc, valid_acc全部绘制在图上并根据时间戳保存
    train_loss = np.array(train_loss)

    valid_loss = np.array(valid_loss)
    train_acc = np.array(train_acc)
    valid_acc = np.array(valid_acc)

    fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)  # 创建一个包含一个axes的figure

    l1 = ax.plot(train_loss, '--', color='blue', label="Train loss")
    l2 = ax.plot(valid_loss, '--', color='red', label="Valid loss")

    ax2 = ax.twinx()
    if np.max(valid_acc) < 0.90:
        ax2.set(ylim=(0.20, 1.00))

    l3 = ax2.plot(train_acc, color='blue', label="Train acc")
    l4 = ax2.plot(valid_acc, color='red', label="Valid acc")

    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')

    EPOCHS = np.array(valid_acc).size
    if EPOCHS == 15:
        blink = 2
    elif EPOCHS == 50:
        blink = 7
    elif EPOCHS == 100:
        blink = 11
    elif EPOCHS == 240:
        blink = 34
    else:
        blink = 18
    ticks = list(range(0, EPOCHS, blink))
    if EPOCHS == 15 or EPOCHS == 50 or EPOCHS == 100:
        ticks.append(EPOCHS - 1)
    else:
        ticks[-1] = EPOCHS

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


def load_Npy(unix_timestamp, Path="./png/"):
    """
    读取训练数据
    :param unix_timestamp:
    :param Path:
    :return:
    """
    npy_path = Path + str(unix_timestamp) + "/"

    train_acc = np.load(npy_path + "train_acc.npy", encoding="latin1")
    valid_acc = np.load(npy_path + "valid_acc.npy", encoding="latin1")

    train_loss = np.load(npy_path + "train_loss.npy", encoding="latin1")
    valid_loss = np.load(npy_path + "valid_loss.npy", encoding="latin1")

    return (train_acc, valid_acc), (train_loss, valid_loss)


def save_Npy(unix_timestamp, train_loss, valid_loss, train_acc, valid_acc, Path="./png/"):
    """
    保存训练数据
    :param unix_timestamp:
    :param train_loss:
    :param valid_loss:
    :param train_acc:
    :param valid_acc:
    :param Path:
    :return:
    """
    npy_path = Path + str(unix_timestamp) + "/"

    np.save(npy_path + "train_acc" + ".npy", train_acc)
    np.save(npy_path + "valid_acc" + ".npy", valid_acc)
    np.save(npy_path + "train_loss" + ".npy", train_loss)
    np.save(npy_path + "valid_loss" + ".npy", valid_loss)


def save_Model(unix_timestamp, model, Path="./png/"):
    """
    根据时间戳unix_timestamp保存model到默认的Path路径下
    :param unix_timestamp:
    :param model:
    :param Path:
    :return:
    """
    # 保存网络模型
    model_path = Path + str(unix_timestamp) + "/"
    torch.save(model, model_path + "model.pt")


def remove_extra_zero(num):
    """
    删除浮点数小数点后多余的0
    :param num:
    :return:
    """
    if isinstance(num, int):
        return num
    if isinstance(num, float):
        num = str(num).rstrip('0')  # 删除小数点后多余的0
        num = int(num.rstrip('.')) if num.endswith('.') else float(num)  # 只剩小数点直接转int，否则转回float
        return num


def get_LearningRate(optimizer):
    """
    获取当前optimizer学习率
    :param optimizer:
    :return:
    """
    return optimizer.state_dict()['param_groups'][0]['lr']


def zip_Dir(dirpath, outputName):
    """
    压缩指定文件夹
    :param dirpath:
    :param outputName:
    :return: 无
    """
    zip = zipfile.ZipFile(outputName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')

        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


def zip_AllData(dirpath):
    """
    压缩dirpath下所有文件，包括文件夹
    :param dirpath:
    :return:
    """
    for dir in os.listdir(dirpath):
        zip_Dir(dirpath + "/" + dir, "./" + dir + ".zip")


def time_Stamp():
    return int(time.time())


def data_Save(model, train_losses, valid_losses, train_acces, valid_acces, unix_timestamp):
    # 绘制训练loss和测试loss、训练acc和测试acc的图
    full_Plot(train_losses, valid_losses, train_acces, valid_acces, unix_timestamp)
    # 保存训练loss和测试loss、训练acc和测试acc
    save_Npy(unix_timestamp, train_losses, valid_losses, train_acces, valid_acces)
    # 保存网络模型
    save_Model(unix_timestamp, model)


def imshow(img):
    img = img / 2 + 0.5
    img = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(img)
